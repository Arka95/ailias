"""Create a new local head-only Character draft from a matched view, without cloud rig/texture requests."""

import json
import math
import os
from pathlib import Path
import sys


def xyz(value):
    return [float(value.x), float(value.y), float(value.z)]


def gnm_to_ue(values, scale):
    x, y, z = values
    return [x*scale, z*scale, y*scale]


def geometry_delta(first, second):
    if not first or len(first) != len(second):
        raise RuntimeError("Native landmark topology changed during commit")
    distances = [math.dist(a, b) for a, b in zip(first, second)]
    if not all(math.isfinite(value) for value in distances):
        raise RuntimeError("Native committed landmarks are nonfinite")
    return {"maximum_distance_cm": max(distances),
            "rms_distance_cm": math.sqrt(sum(value*value for value in distances)/len(distances))}


def export_saved_geometry(character, subsystem, unreal, destination, output):
    """Evaluate the committed Character surface, independently of sculpt-control positions."""
    parameters = unreal.MetaHumanGeometryExportParams()
    parameters.project_path = destination
    parameters.head_skeletal_mesh = True
    parameters.body_skeletal_mesh = False
    parameters.full_body_skeletal_mesh = False
    parameters.overwrite_existing_assets = False
    unreal.MetaHumanCharacterExportBlueprintLibrary.export_geometry(character, parameters)
    meshes = [unreal.load_asset(path) for path in unreal.EditorAssetLibrary.list_assets(
        destination, recursive=True, include_folder=False)]
    meshes = [mesh for mesh in meshes if isinstance(mesh, unreal.SkeletalMesh)]
    if len(meshes) != 1:
        raise RuntimeError("Persistence export requires exactly one native head mesh")
    vertices, triangles = subsystem.get_mesh_data_for_conforming(meshes[0])
    points = [xyz(point) for point in vertices]
    if (not points or not triangles or len(triangles) % 3
            or not all(math.isfinite(value) for point in points for value in point)):
        raise RuntimeError("Persistence export produced empty or nonfinite head geometry")
    payload = {"vertices": points, "indices": list(triangles),
               "coordinate_frame": "native committed head skeletal mesh; centimetres"}
    Path(output).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return {"path": str(output), "mesh": meshes[0].get_path_name(),
            "vertex_count": len(points), "triangle_count": len(triangles)//3}


def run(request, unreal, publish):
    result = {
        "status": "running", "stage": "load_approved_custom_head",
        "cloud_requested": False, "rig_requested": False,
        "assembled": False, "likeness_verified": False, "appearance_transferred": False,
        "engine": unreal.SystemLibrary.get_engine_version(),
    }

    def stage(value):
        result["stage"] = value
        publish(result)

    if not result["engine"].startswith("5.8."):
        raise ValueError("Custom-head adapter targets the verified UE 5.8 API")
    if unreal.EditorAssetLibrary.list_assets(request["destination"], recursive=True, include_folder=False):
        raise ValueError("Character destination is occupied; no overwrite is allowed")
    mesh = unreal.load_asset(request["mesh"])
    if not isinstance(mesh, unreal.StaticMesh):
        raise ValueError("Custom head must be a saved static mesh")
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    topology = subsystem.get_mesh_data_for_conforming(mesh)
    if not topology or not topology[0] or not topology[1]:
        raise ValueError("Custom head has no conformable topology")
    result["target_geometry"] = {"vertices": len(topology[0]), "triangles": len(topology[1])//3}
    stage("track_perspective_face_contours")
    camera = json.loads(Path(request["camera"]).read_text())
    width, height = camera["image_size"]
    pixels = Path(request["rgba"]).read_bytes()
    if len(pixels) != width*height*4 or camera["projection"] != "perspective":
        raise ValueError("Tracking pixels must match the recorded perspective camera")
    colours = [unreal.Color(*pixels[index:index+4]) for index in range(0, len(pixels), 4)]
    curves = subsystem.track_face_landmarks_from_image(colours, width, height)
    if not curves:
        raise RuntimeError("Native face contour tracking failed; no Character solve was claimed")
    result["tracked_curves"] = len(curves)
    result["contours"] = {str(name): [[float(p.x), float(p.y)] for p in points.tracking_points]
                           for name, points in curves.items()}
    stage("create_local_character")
    if request.get("source_character"):
        source_character = unreal.load_asset(request["source_character"])
        if (not isinstance(source_character, unreal.MetaHumanCharacter)
                or subsystem.is_object_added_for_editing(source_character)):
            raise ValueError("Native refinement source must be an existing unedited Character")
        character = unreal.EditorAssetLibrary.duplicate_asset(
            request["source_character"], request["destination"] + "/HeadOnlyDraft")
        result["source_character"] = request["source_character"]
    else:
        factory = unreal.MetaHumanCharacterFactoryNew()
        factory.edit_after_new = False
        character = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "HeadOnlyDraft", request["destination"], unreal.MetaHumanCharacter, factory)
    if not isinstance(character, unreal.MetaHumanCharacter):
        raise RuntimeError("Character factory did not return a MetaHuman Character")
    result["character"] = character.get_path_name()
    stage("register_character_for_editing")
    registered = subsystem.try_add_object_to_edit(character)
    if not registered:
        raise RuntimeError("Character could not enter editing mode")
    try:
        before = [xyz(point) for point in subsystem.get_face_landmarks(character)]
        if request.get("source_character") and character.get_editor_property("has_face_dna_blendshapes"):
            stage("remove_inherited_face_rig")
            subsystem.remove_face_rig(character)
            if character.get_editor_property("has_face_dna_blendshapes"):
                raise RuntimeError("Inherited face rig must be removed before native geometry editing")
            result["inherited_face_rig_removed"] = True
        target = unreal.ConformTargetMesh()
        target.target_parts_type = unreal.TargetPartsType.HEAD_ONLY
        # The saved static preview faces +X; the native Character template faces +Y.
        target.head_vertices = [unreal.Vector3f(-v.y, v.x, v.z) for v in topology[0]]
        target.head_vertex_indices = topology[1]
        result["target_basis"] = "MetaHuman: (GNM X,Z,Y)*100; original static preview unchanged"
        parameters = unreal.ConformTargetParams()
        parameters.conform_target_mesh = target
        parameters.auto_solve = True
        parameters.curve_tracking_points = curves
        parameters.image_size = unreal.IntPoint(width, height)
        view = unreal.MinimalViewInfo()
        view.location = unreal.Vector(*gnm_to_ue(camera["position_gnm"], 100))
        forward = unreal.Vector(*gnm_to_ue(camera["forward_gnm"], 1))
        up = unreal.Vector(*gnm_to_ue(camera["up_gnm"], 1))
        view.rotation = unreal.MathLibrary.make_rot_from_xz(forward, up)
        view.fov = camera["horizontal_fov_degrees"]
        view.aspect_ratio = width/height
        view.projection_mode = unreal.CameraProjectionMode.PERSPECTIVE
        parameters.camera_view_info = view
        settings = parameters.body_conform_solve_settings
        settings.pipeline_name = "head_only"
        parameters.body_conform_solve_settings = settings
        key = unreal.MetaHumanCharacterTargetMeshKey()
        key.head_mesh = mesh
        stage("native_head_only_auto_solve")
        if not subsystem.conform_to_target_meshes(character, key, parameters):
            raise RuntimeError("Native head-only solver returned false")
        result["face_landmarks_before"] = before
        result["face_landmarks_solved_in_target_pose"] = [
            xyz(point) for point in subsystem.get_face_landmarks(character)]
        stage("finalize_native_conform_as_a_pose")
        subsystem.commit_posed_state_as_a_pose(character, key)
        result["face_landmarks_after"] = [xyz(point) for point in subsystem.get_face_landmarks(character)]
        result["pose_finalization_delta"] = geometry_delta(
            result["face_landmarks_solved_in_target_pose"], result["face_landmarks_after"])
        stage("commit_and_save_conformed_character")
        subsystem.commit_face_state(character)
        result["face_landmarks_face_committed"] = [
            xyz(point) for point in subsystem.get_face_landmarks(character)]
        subsystem.commit_body_state(character)
        result["face_landmarks_body_committed"] = [
            xyz(point) for point in subsystem.get_face_landmarks(character)]
        result["commit_geometry"] = {
            "face": geometry_delta(result["face_landmarks_after"], result["face_landmarks_face_committed"]),
            "body": geometry_delta(result["face_landmarks_face_committed"],
                                   result["face_landmarks_body_committed"]),
        }
        publish(result)
        if any(value["maximum_distance_cm"] > .0001 for value in result["commit_geometry"].values()):
            raise RuntimeError("Native commit changed the conformed geometry; inspect commit_geometry")
        if not unreal.EditorAssetLibrary.save_loaded_asset(character, only_if_is_dirty=False):
            raise RuntimeError("Conformed Character package could not be saved")
        result["face_landmarks_saved"] = [
            xyz(point) for point in subsystem.get_face_landmarks(character)]
        result["save_geometry"] = geometry_delta(
            result["face_landmarks_body_committed"], result["face_landmarks_saved"])
        if result["save_geometry"]["maximum_distance_cm"] > .0001:
            publish(result)
            raise RuntimeError("Native save changed the committed geometry; inspect save_geometry")
        result["geometry_reference"] = "post face/body commits and successful package save; native centimetres"
        if request.get("geometry_output"):
            stage("export_committed_geometry_reference")
            result["saved_surface"] = export_saved_geometry(
                character, subsystem, unreal, request["destination"] + "/SavedSurface",
                request["geometry_output"])
            result["face_landmarks_after_export"] = [
                xyz(point) for point in subsystem.get_face_landmarks(character)]
            result["export_landmark_delta"] = geometry_delta(
                result["face_landmarks_saved"], result["face_landmarks_after_export"])
            publish(result)
        result.update(status="head_conformed_awaiting_review", stage="review_conformed_shape",
                      assets=[character.get_path_name()])
    finally:
        primary = sys.exception()
        try:
            subsystem.remove_object_to_edit(character)
        except Exception as error:
            result["cleanup_error"] = type(error).__name__
            if primary is None:
                raise
            primary.add_note("Character editing cleanup also failed: " + type(error).__name__)
            try:
                publish(result)
            except Exception:
                print("Could not publish native cleanup diagnostics", file=sys.stderr)
    return result


def main():
    import unreal

    request = json.loads(Path(os.environ["AILIAS_CONFORM_REQUEST"]).read_text())
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
