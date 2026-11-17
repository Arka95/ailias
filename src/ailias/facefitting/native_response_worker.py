"""Measure native sculpt responses on a disposable copy, never on the source."""

from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def coordinates(values):
    return [[float(v.x), float(v.y), float(v.z)] for v in values]


def maximum_distance(first, second):
    if not first or len(first) != len(second):
        raise ValueError("Native response topology changed")
    result = max(math.dist(a, b) for a, b in zip(first, second))
    if not math.isfinite(result):
        raise ValueError("Nonfinite native response")
    return result


def write_float32(stream, points):
    values = array("f", (value for point in points for value in point))
    if sys.byteorder != "little":
        values.byteswap()
    stream.write(values.tobytes())


def run(request, unreal, publish):
    step = request["step_cm"]
    if (isinstance(step, bool) or not isinstance(step, (int, float))
            or not math.isfinite(step) or not .001 <= step <= .25):
        raise ValueError("Native response step must be between .001 and .25 cm")
    root = Path(request["output"])
    result = {"status": "running", "source_saved": False, "likeness_verified": False,
              "step_cm": step, "responses": [], "format": "little-endian-float32"}
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    if unreal.EditorAssetLibrary.list_assets(
            request["destination"], recursive=True, include_folder=False):
        raise ValueError("Native response namespace is occupied")
    character = unreal.EditorAssetLibrary.duplicate_asset(
        request["character"], request["destination"] + "/ResponseDraft")
    if not isinstance(character, unreal.MetaHumanCharacter):
        raise RuntimeError("Cannot create isolated native response Character")
    if character.get_editor_property("has_face_dna_blendshapes"):
        raise ValueError("Native response requires an unrigged source")
    if not subsystem.try_add_object_to_edit(character):
        raise RuntimeError("Cannot register isolated native response Character")
    coefficients = None

    def stage(name):
        result["stage"] = name
        publish(result)

    def landmarks():
        return coordinates(subsystem.get_face_landmarks(character))

    def export():
        # Export reads committed Character state, not the live sculpt preview.
        subsystem.commit_face_state(character)
        params = unreal.MetaHumanGeometryExportParams()
        params.project_path = request["destination"] + "/Mesh"
        params.head_skeletal_mesh = True
        params.body_skeletal_mesh = False
        params.full_body_skeletal_mesh = False
        params.overwrite_existing_assets = True
        unreal.MetaHumanCharacterExportBlueprintLibrary.export_geometry(character, params)
        assets = unreal.EditorAssetLibrary.list_assets(
            params.project_path, recursive=True, include_folder=False)
        meshes = [unreal.load_asset(asset) for asset in assets]
        meshes = [mesh for mesh in meshes if isinstance(mesh, unreal.SkeletalMesh)]
        if len(meshes) != 1:
            raise RuntimeError("Expected one native response head mesh")
        vertices, triangles = subsystem.get_mesh_data_for_conforming(meshes[0])
        values = coordinates(vertices)
        if not values or not triangles or not all(math.isfinite(v) for p in values for v in p):
            raise RuntimeError("Native response mesh is empty or nonfinite")
        return values, list(triangles)

    try:
        stage("read_native_reference")
        original = landmarks()
        coefficients = list(subsystem.get_face_model_coefficients(character))
        if not coefficients or not all(math.isfinite(value) for value in coefficients):
            raise RuntimeError("Missing native state coefficients")
        # This is an opaque state snapshot, not an array of independent shape modes.
        subsystem.set_face_model_coefficients(character, coefficients)
        if maximum_distance(original, landmarks()) > 1e-4:
            raise RuntimeError("Native state roundtrip changed the reference shape")
        indices = request.get("indices")
        if indices is None:
            indices = list(range(len(original)))
        if (not isinstance(indices, list) or not indices
                or any(type(i) is not int or not 0 <= i < len(original) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("Native response indices must be unique existing controls")
        stage("export_reference")
        baseline, triangles = export()
        result.update(landmark_count=len(original), vertex_count=len(baseline),
                      reference_landmarks=original, indices=indices)
        with (root / "reference.f32").open("xb") as stream:
            write_float32(stream, baseline)
        (root / "triangles.json").write_text(json.dumps(triangles), encoding="utf-8")
        with (root / "responses.f32").open("xb") as stream:
            for index in indices:
                for axis in range(3):
                    stage(f"probe_control_{index}_axis_{axis}")
                    started = time.monotonic()
                    subsystem.set_face_model_coefficients(character, coefficients)
                    if maximum_distance(original, landmarks()) > 1e-4:
                        raise RuntimeError("Native reference restoration failed")
                    delta = [0., 0., 0.]
                    delta[axis] = step
                    subsystem.translate_face_landmarks(character, [index], [unreal.Vector(*delta)])
                    changed, topology = export()
                    if topology != triangles or len(changed) != len(baseline):
                        raise RuntimeError("Native response changed canonical mesh topology")
                    derivative = [[(b-a)/step for a, b in zip(first, second)]
                                  for first, second in zip(baseline, changed)]
                    write_float32(stream, derivative)
                    stream.flush()
                    result["responses"].append({
                        "index": index, "axis": axis,
                        "maximum_mesh_displacement_cm": maximum_distance(baseline, changed),
                        "seconds": time.monotonic()-started,
                    })
        stage("verify_whole_mesh_restoration")
        subsystem.set_face_model_coefficients(character, coefficients)
        restored, topology = export()
        result["restored_mesh_displacement_cm"] = maximum_distance(baseline, restored)
        if (topology != triangles or result["restored_mesh_displacement_cm"] > 1e-4
                or maximum_distance(original, landmarks()) > 1e-4):
            raise RuntimeError("Whole native mesh restoration failed")
        result.update(status="native_response_measured", stage="awaiting_geometry_fit")
        return result
    finally:
        try:
            if coefficients is not None:
                subsystem.set_face_model_coefficients(character, coefficients)
                subsystem.commit_face_state(character)
        finally:
            subsystem.remove_object_to_edit(character)


def main():
    import unreal

    request = json.loads(Path(os.environ["AILIAS_NATIVE_RESPONSE_REQUEST"]).read_text())
    receipt = Path(request["receipt"])
    source = Path(request["source_package"])
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    def publish(value):
        receipt.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

    result = {}
    try:
        result = run(request, unreal, publish)
    except Exception as error:
        result = json.loads(receipt.read_text()) if receipt.is_file() else {}
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        result["source_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == before
        if not result["source_unchanged"]:
            result.update(status="failed", error="Native response source changed")
        publish(result)
        unreal.SystemLibrary.quit_editor()


if __name__ == "__main__":
    main()
