"""Export genuine native skin UVs and import photo textures in an isolated Unreal process."""

import json
import os
from pathlib import Path


def export_skin_uv_mesh(request, unreal):
    mesh = unreal.load_asset(request["mesh"])
    if not isinstance(mesh, unreal.SkeletalMesh):
        raise ValueError("Native UV export requires the newly exported head skeletal mesh")
    materials = mesh.get_editor_property("materials")
    skin_slots = []
    for index, slot in enumerate(materials):
        label = str(slot.material_slot_name).lower()
        if slot.material_interface is not None:
            label += " " + slot.material_interface.get_path_name().rsplit("/", 1)[-1].lower()
        if "head" in label or "skin" in label:
            skin_slots.append(index)
    if not skin_slots:
        raise RuntimeError("The native mesh does not identify a skin material section")
    dynamic = unreal.DynamicMesh()
    _, outcome = unreal.GeometryScript_AssetUtils.copy_mesh_from_skeletal_mesh(
        mesh, dynamic, unreal.GeometryScriptCopyMeshFromAssetOptions(),
        unreal.GeometryScriptMeshReadLOD())
    if outcome != unreal.GeometryScriptOutcomePins.SUCCESS:
        raise RuntimeError("Cannot copy native geometry with its original UV layers")
    queries = unreal.GeometryScript_MeshQueries
    vertices, triangles, uvs, index_map = [], [], [], {}
    for triangle_id in range(queries.get_num_triangle_i_ds(dynamic)):
        material, valid = unreal.GeometryScript_Materials.get_triangle_material_id(dynamic, triangle_id)
        if not valid or material not in skin_slots:
            continue
        indices, valid = queries.get_triangle_indices(dynamic, triangle_id)
        positions_valid, *positions = queries.get_triangle_positions(dynamic, triangle_id)
        *corners, uvs_valid = queries.get_triangle_u_vs(dynamic, 0, triangle_id)
        if not valid or not positions_valid or not uvs_valid:
            raise RuntimeError("Native skin triangle is missing geometry or UVs")
        mapped = []
        for index, position in zip((indices.x, indices.y, indices.z), positions):
            if index not in index_map:
                index_map[index] = len(vertices)
                vertices.append([float(position.x), float(position.y), float(position.z)])
            mapped.append(index_map[index])
        triangles.append(mapped)
        uvs.append([[float(value.x), float(value.y)] for value in corners])
    if not triangles:
        raise RuntimeError("No UV-mapped native skin triangles were exported")
    output = Path(request["output"])
    output.write_text(json.dumps({
        "vertices": vertices, "triangles": triangles, "triangle_uvs": uvs,
        "coordinate_frame": "canonical native skeletal mesh centimetres",
        "canonical_to_target": request.get("canonical_to_target"),
        "material_slots": skin_slots, "source_mesh": request["mesh"],
    }, allow_nan=False), encoding="utf-8")
    return {"status": "native_skin_uvs_exported", "output": str(output),
            "vertex_count": len(vertices), "triangle_count": len(triangles),
            "native_uvs_preserved": True, "source_mesh_modified": False}


def import_base_color_texture(request, unreal):
    library = unreal.EditorAssetLibrary
    if library.list_assets(request["destination"], recursive=True, include_folder=False):
        raise ValueError("Texture destination is occupied; refusing to overwrite assets")
    task = unreal.AssetImportTask()
    task.filename = request["image"]
    task.destination_path = request["destination"]
    task.destination_name = "PhotoBaseColor"
    task.automated = True
    task.replace_existing = False
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    textures = [unreal.load_asset(path) for path in task.imported_object_paths]
    textures = [texture for texture in textures if isinstance(texture, unreal.Texture2D)]
    if len(textures) != 1:
        raise RuntimeError("Photo texture import did not produce exactly one Texture2D")
    texture = textures[0]
    texture.set_editor_property("srgb", True)
    texture.set_editor_property("virtual_texture_streaming", False)
    if not library.save_loaded_asset(texture, only_if_is_dirty=False):
        raise RuntimeError("Imported native photo texture could not be saved")
    return {"status": "native_photo_texture_imported", "texture": texture.get_path_name()}


def main():
    import unreal

    request = json.loads(Path(os.environ["AILIAS_FRESH_VIDEO_REQUEST"]).read_text())
    receipt = Path(request["receipt"])
    try:
        operations = {"export_uv": export_skin_uv_mesh, "import_texture": import_base_color_texture}
        if request["operation"] not in operations:
            raise ValueError("Unsupported native surface operation")
        result = operations[request["operation"]](request, unreal)
    except Exception as error:
        receipt.write_text(json.dumps({
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "project_may_contain_partial_outputs": True,
        }), encoding="utf-8")
        raise
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
