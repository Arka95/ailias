"""Import a fresh fitted head and create a new native Character, never clone one."""

import importlib.util
import json
import math
import os
from pathlib import Path
import sys


def load_worker_module(path):
    spec = importlib.util.spec_from_file_location("fresh_conform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_character_request(request, unreal, publish=lambda value: None):
    if request["operation"] == "verify":
        character = unreal.load_asset(request["character"])
        if not isinstance(character, unreal.MetaHumanCharacter):
            raise RuntimeError("The newly saved Character cannot be independently reloaded")
        subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
        if not subsystem.try_add_object_to_edit(character):
            raise RuntimeError("The newly saved Character cannot enter editing mode")
        try:
            points = [[float(v.x), float(v.y), float(v.z)]
                      for v in subsystem.get_face_landmarks(character)]
            expected = request["expected"]
            result = {"status": "verifying", "character": request["character"],
                      "landmarks": points, "expected_landmarks": expected, "likeness_verified": False,
                      "reference": "post-commit saved state; native centimetres",
                      "maximum_allowed_distance_cm": .0001}
            publish(result)
            if (not points or len(points) != len(expected)
                    or any(len(point) != 3 for point in expected)):
                raise RuntimeError("Reloaded Character landmark topology differs from saved state")
            distances = [math.dist(a, b) for a, b in zip(points, expected)]
            if not all(math.isfinite(distance) for distance in distances):
                raise RuntimeError("Reloaded Character landmarks are nonfinite")
            result.update(maximum_distance_cm=max(distances),
                          rms_distance_cm=math.sqrt(sum(d*d for d in distances)/len(distances)))
            publish(result)
            if request.get("expected_surface"):
                module = load_worker_module(request["conform_worker"])
                output = Path(request["expected_surface"]).with_name("native-geometry-reloaded.json")
                result["reloaded_surface"] = module.export_saved_geometry(
                    character, subsystem, unreal, request["surface_destination"], output)
                expected_surface = json.loads(Path(request["expected_surface"]).read_text(encoding="utf-8"))
                surface = json.loads(output.read_text(encoding="utf-8"))
                if surface["indices"] != expected_surface["indices"]:
                    raise RuntimeError("Reloaded native head surface topology changed")
                result["surface_delta"] = module.geometry_delta(
                    expected_surface["vertices"], surface["vertices"])
                publish(result)
                if result["surface_delta"]["maximum_distance_cm"] > .0001:
                    raise RuntimeError(
                        "Reloaded native head surface changed: "
                        f"{result['surface_delta']['maximum_distance_cm']:.9g} cm maximum")
                if request.get("native_uv_output"):
                    result["stage"] = "export_native_skin_uvs"
                    publish(result)
                    uv_worker = load_worker_module(request["surface_worker"])
                    result["native_skin_uvs"] = uv_worker.export_skin_uv_mesh({
                        "mesh": result["reloaded_surface"]["mesh"],
                        "output": request["native_uv_output"],
                        "canonical_to_target": request["canonical_to_target"],
                    }, unreal)
            elif result["maximum_distance_cm"] > .0001:
                raise RuntimeError(
                    "Reloaded Character geometry differs from saved state: "
                    f"{result['maximum_distance_cm']:.9g} cm maximum")
            result["status"] = "fresh_character_reloaded"
            return result
        finally:
            primary = sys.exception()
            try:
                subsystem.remove_object_to_edit(character)
            except Exception as error:
                if primary is None:
                    raise
                primary.add_note("Character editing cleanup also failed: " + type(error).__name__)

    destination = request["destination"]
    if unreal.EditorAssetLibrary.list_assets(destination, recursive=True, include_folder=False):
        raise ValueError("Fresh destination is occupied; refusing to reuse or overwrite assets")
    task = unreal.AssetImportTask()
    task.filename = request["obj"]
    task.destination_path = destination + "/Input"
    task.destination_name = "FreshFittedHead"
    task.automated = True
    task.replace_existing = False
    task.save = True
    options = unreal.FbxImportUI()
    options.import_mesh = True
    options.import_as_skeletal = False
    options.mesh_type_to_import = unreal.FBXImportType.FBXIT_STATIC_MESH
    options.import_materials = False
    options.import_textures = False
    options.static_mesh_import_data.combine_meshes = True
    task.options = options
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    meshes = [name for name in unreal.EditorAssetLibrary.list_assets(
        task.destination_path, recursive=True, include_folder=False)
        if isinstance(unreal.load_asset(name), unreal.StaticMesh)]
    if len(meshes) != 1:
        raise RuntimeError("Fresh geometry import did not produce exactly one static head mesh")
    bounds = unreal.load_asset(meshes[0]).get_bounding_box()
    actual_bounds = [[float(v.x), float(v.y), float(v.z)] for v in (bounds.min, bounds.max)]
    if not all(math.isclose(a, b, abs_tol=.001, rel_tol=0)
               for actual, expected in zip(actual_bounds, request["expected_bounds"])
               for a, b in zip(actual, expected)):
        raise RuntimeError("Fresh OBJ import axes or scale differ from the existing handoff contract")
    module = load_worker_module(request["conform_worker"])
    conformed = module.run({
        "mesh": meshes[0], "destination": destination + "/Character",
        "camera": request["camera"], "rgba": request["rgba"],
        "geometry_output": request.get("geometry_output"),
    }, unreal, publish)
    conformed.update(imported_mesh=meshes[0], previous_character_reused=False)
    return conformed


def main(request=None, *, keep_alive=False):
    import unreal

    if request is None:
        request = json.loads(Path(os.environ["AILIAS_FRESH_VIDEO_REQUEST"]).read_text())
    receipt = Path(request["receipt"])

    def publish(value):
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(receipt)

    try:
        result = execute_character_request(request, unreal, publish)
    except Exception as error:
        try:
            result = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
        except (OSError, ValueError):
            result = {}
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      project_may_contain_partial_outputs=True)
        publish(result)
        raise
    publish(result)


if __name__ == "__main__":
    main()
