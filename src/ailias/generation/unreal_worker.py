"""UE 5.8 commandlet boundary; an imported mesh is never a MetaHuman success."""

import json
import os
from pathlib import Path


def run(request, unreal):
    version = unreal.SystemLibrary.get_engine_version()
    if not version.startswith("5.8."):
        raise RuntimeError("The installed editor API must match the UE 5.8 adapter")
    result = {"engine_version": version, "assembled": False, "rig_verified": False,
              "likeness_verified": False, "completed_operations": []}
    if request["operation"] == "import":
        destination = request["destination"]
        if unreal.EditorAssetLibrary.list_assets(destination, recursive=True, include_folder=False):
            raise RuntimeError("Destination already contains assets; no overwrite is allowed")
        task = unreal.AssetImportTask()
        task.filename = request["mesh"]
        task.destination_path = destination
        task.destination_name = "HeadCandidate"
        task.automated = True
        task.replace_existing = False
        task.save = True
        options = unreal.FbxImportUI()
        options.import_mesh = True
        options.import_as_skeletal = False
        options.mesh_type_to_import = unreal.FBXImportType.FBXIT_STATIC_MESH
        options.import_textures = True
        options.import_materials = True
        options.static_mesh_import_data.combine_meshes = True
        task.options = options
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        paths = unreal.EditorAssetLibrary.list_assets(destination, recursive=True, include_folder=False)
        assets = [unreal.load_asset(path) for path in paths]
        if (not any(isinstance(asset, unreal.StaticMesh) for asset in assets)
                or not any(isinstance(asset, unreal.Texture2D) for asset in assets)):
            raise RuntimeError("A textured static mesh was not imported; inspect the editor log")
        result.update(status="mesh_imported", assets=paths, metric_scale_verified=False,
                      requires_manual_identity_solve=True)
        result["completed_operations"].append("textured_static_mesh_import")
        return result

    character = unreal.load_asset(request["character"])
    identity = unreal.load_asset(request["identity"])
    if not isinstance(character, unreal.MetaHumanCharacter) or not isinstance(identity, unreal.MetaHumanIdentity):
        raise RuntimeError("Provide existing MetaHuman Character and solved MetaHuman Identity assets")
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    if not subsystem.try_add_object_to_edit(character):
        raise RuntimeError("Could not register this character for editing")
    try:
        parameters = unreal.ImportFromIdentityParams()
        parameters.use_metric_scale = False
        code = subsystem.import_from_identity(character, identity, parameters)
        if code != unreal.ImportErrorCode.SUCCESS:
            raise RuntimeError(f"Identity conform failed: {code}")
        subsystem.commit_face_state(character)
        if not unreal.EditorAssetLibrary.save_loaded_asset(character, only_if_is_dirty=False):
            raise RuntimeError("Conformed character could not be saved")
        result.update(status="identity_conformed", character=request["character"],
                      identity=request["identity"])
        result["completed_operations"].append("identity_conform")
        if request["cloud_services"]:
            rig = unreal.MetaHumanCharacterAutoRiggingRequestParams()
            rig.blocking, rig.report_progress = True, False
            rig.rig_type = unreal.MetaHumanRigType.JOINTS_AND_BLENDSHAPES
            rig_result = subsystem.request_auto_rigging(character, rig)
            if rig_result is False:
                raise RuntimeError("Epic cloud auto-rig request failed")
            texture = unreal.MetaHumanCharacterTextureRequestParams()
            texture.blocking, texture.report_progress = True, False
            texture_result = subsystem.request_texture_sources(character, texture)
            if texture_result is False:
                raise RuntimeError("Epic cloud texture request failed")
            if not unreal.EditorAssetLibrary.save_loaded_asset(character, only_if_is_dirty=False):
                raise RuntimeError("Character changes could not be saved")
            result.update(status="identity_conformed_cloud_requests_returned",
                          rig_api_result=str(rig_result), texture_api_result=str(texture_result))
            result["completed_operations"].extend(["blocking_cloud_rig_request", "blocking_cloud_texture_request"])
        return result
    finally:
        subsystem.remove_object_to_edit(character)


def main():
    import unreal

    request = json.loads(Path(os.environ["AILIAS_UNREAL_REQUEST"]).read_text(encoding="utf-8"))
    receipt = Path(request["receipt"])
    try:
        result = run(request, unreal)
    except Exception as error:
        receipt.write_text(json.dumps({
            "status": "failed", "error_type": type(error).__name__,
            "project_may_contain_partial_outputs": True,
            "assembled": False, "rig_verified": False,
        }, indent=2), encoding="utf-8")
        raise
    receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
