"""Import a wrapped native template and persistent photo material into a new Character, without cloud work."""

import json
import os
from pathlib import Path


def run(request, unreal, publish):
    result = {"status": "running", "stage": "import_native_template",
              "cloud_requested": False, "rig_requested": False, "likeness_approved": False}

    def stage(value):
        result["stage"] = value
        publish(result)

    if unreal.EditorAssetLibrary.list_assets(request["destination"], recursive=True, include_folder=False):
        raise ValueError("Recovery destination already contains assets")
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    mesh_task = unreal.AssetImportTask()
    mesh_task.filename = request["template"]
    mesh_task.destination_path = request["destination"]
    mesh_task.destination_name = "WrappedNativeHead"
    mesh_task.automated = True
    mesh_task.save = True
    mesh_task.factory = unreal.FbxFactory()
    options = unreal.FbxImportUI()
    options.import_as_skeletal = False
    options.import_materials = False
    options.import_textures = False
    options.mesh_type_to_import = unreal.FBXImportType.FBXIT_STATIC_MESH
    options.static_mesh_import_data.combine_meshes = True
    options.static_mesh_import_data.generate_lightmap_u_vs = False
    mesh_task.options = options
    colour_task = unreal.AssetImportTask()
    colour_task.filename = request["texture"]
    colour_task.destination_path = request["destination"]
    colour_task.destination_name = "PhotoColour"
    colour_task.automated = True
    colour_task.save = True
    tools.import_asset_tasks([mesh_task, colour_task])
    meshes = [unreal.load_asset(path) for path in mesh_task.imported_object_paths]
    meshes = [mesh for mesh in meshes if isinstance(mesh, unreal.StaticMesh)]
    textures = [unreal.load_asset(path) for path in colour_task.imported_object_paths]
    textures = [texture for texture in textures if isinstance(texture, unreal.Texture2D)]
    if len(meshes) != 1 or len(textures) != 1:
        raise ValueError("Require one genuine native-topology static head and one photo texture")
    mesh, texture = meshes[0], textures[0]
    stage("create_persistent_photo_material")
    material = tools.create_asset("PhotoMaterial", request["destination"], unreal.Material,
                                  unreal.MaterialFactoryNew())
    editing = unreal.MaterialEditingLibrary
    sample = editing.create_material_expression(material, unreal.MaterialExpressionTextureSampleParameter2D)
    sample.set_editor_property("parameter_name", "PhotoColour")
    sample.set_editor_property("texture", texture)
    editing.connect_material_property(sample, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    roughness = editing.create_material_expression(material, unreal.MaterialExpressionConstant)
    roughness.set_editor_property("r", .6)
    editing.connect_material_property(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    editing.recompile_material(material)
    instance = tools.create_asset("PhotoMaterialInstance", request["destination"],
                                  unreal.MaterialInstanceConstant,
                                  unreal.MaterialInstanceConstantFactoryNew())
    editing.set_material_instance_parent(instance, material)
    editing.set_material_instance_texture_parameter_value(instance, "PhotoColour", texture)
    stage("fit_native_uv_corresponding_template")
    character = unreal.EditorAssetLibrary.duplicate_asset(
        request["character"], request["destination"] + "/PhotoHead")
    if not isinstance(character, unreal.MetaHumanCharacter):
        raise ValueError("Could not duplicate the prior Character without changing it")
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    if not subsystem.try_add_object_to_edit(character):
        raise ValueError("New Character cannot enter editing mode")
    try:
        params = unreal.ImportFromTemplateParams()
        params.match_vertices_by_u_vs = True
        params.use_eye_meshes = False
        params.use_teeth_mesh = False
        params.isolate_head_from_body = True
        params.alignment_options = unreal.MetaHumanAlignmentOptions.NONE
        code = subsystem.import_from_template(character, mesh, None, None, None, params)
        result["import_code"] = str(code)
        if code.name.upper() != "SUCCESS":
            raise RuntimeError("Native UV-corresponding template import failed: " + str(code))
        stage("persist_character_material_override")
        settings = character.get_editor_property("skin_settings")
        overrides = settings.texture_material_overrides
        overrides.enable_material_overrides = True
        overrides.inherit_ui_params_and_src_textures = False
        materials = overrides.material_overrides
        slots = unreal.MetaHumanCharacterSkinMaterialSlot
        materials.skin = {getattr(slots, slot): instance for slot in
                          ("LOD0", "LOD1", "LOD2", "LOD3", "LOD4", "LOD5TO7")}
        overrides.material_overrides = materials
        settings.texture_material_overrides = overrides
        character.set_editor_property("skin_settings", settings)
        subsystem.commit_face_state(character)
        for asset in (material, instance, texture, mesh, character):
            if not unreal.EditorAssetLibrary.save_loaded_asset(asset, only_if_is_dirty=False):
                raise RuntimeError("Could not save an actual recovery asset")
        result["character"] = character.get_path_name()
        result["material"] = instance.get_path_name()
        result["assets"] = [asset.get_path_name() for asset in (material, instance, texture, mesh, character)]
    finally:
        subsystem.remove_object_to_edit(character)
    stage("verify_reload_and_export_evaluated_face")
    if not subsystem.try_add_object_to_edit(character):
        raise RuntimeError("Saved Character cannot reload for verification")
    actor = None
    try:
        actor = subsystem.spawn_meta_human_actor(character, True)
        face = next(c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent)
                    if c.get_name() == "Face")
        face_mesh = face.get_skeletal_mesh_asset()
        result["skin_material"] = face.get_material(0).get_path_name()
        result["photo_material_active"] = instance.get_name() in result["skin_material"]
        if not result["photo_material_active"]:
            raise RuntimeError("Saved photo material did not activate on the native head")
        export = unreal.AssetExportTask()
        export.object = face_mesh
        export.filename = request["native_fbx"]
        export.automated = True
        export.prompt = False
        export.options = unreal.FbxExportOption()
        export.options.level_of_detail = False
        export.options.export_morph_targets = False
        if not unreal.Exporter.run_asset_export_task(export):
            raise RuntimeError("Could not export the evaluated native face for comparison")
        result.update(status="native_photo_head_awaiting_likeness_review",
                      stage="compare_actual_surface_and_appearance")
    finally:
        if actor:
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
        subsystem.remove_object_to_edit(character)
    return result


def main():
    import unreal

    request = json.loads(Path(os.environ["AILIAS_LIKENESS_REQUEST"]).read_text())
    receipt = Path(request["receipt"])

    def publish(value):
        receipt.write_text(json.dumps(value, indent=2), encoding="utf-8")

    try:
        result = run(request, unreal, publish)
    except Exception as error:
        result = json.loads(receipt.read_text()) if receipt.is_file() else {}
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      project_may_contain_partial_outputs=True)
        publish(result)
        raise
    publish(result)


if __name__ == "__main__":
    main()
