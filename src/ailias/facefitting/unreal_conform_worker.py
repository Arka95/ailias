"""Create a new local head-only Character draft from a matched view, without cloud rig/texture requests."""

import json
import os
from pathlib import Path


def xyz(value):
    return [float(value.x), float(value.y), float(value.z)]


def gnm_to_ue(values, scale):
    x, y, z = values
    return [x*scale, z*scale, y*scale]


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
        result["face_landmarks_after"] = [xyz(point) for point in subsystem.get_face_landmarks(character)]
        stage("commit_and_save_conformed_character")
        subsystem.commit_face_state(character)
        subsystem.commit_body_state(character)
        if not unreal.EditorAssetLibrary.save_loaded_asset(character, only_if_is_dirty=False):
            raise RuntimeError("Conformed Character package could not be saved")
        result.update(status="head_conformed_awaiting_review", stage="review_conformed_shape",
                      assets=[character.get_path_name()])
    finally:
        subsystem.remove_object_to_edit(character)
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
