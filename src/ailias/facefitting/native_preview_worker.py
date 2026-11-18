"""Render the freshly reloaded editor-native Character, without cloud rigging or old screenshots."""

import json
import math
import os
from pathlib import Path
import time


def face_framing(points, width=540, height=960, distance=75):
    """Frame native face controls rather than skeletal component bounds, which include shoulders."""
    values = [[float(v.x), float(v.y), float(v.z)] for v in points]
    if len(values) < 3 or not all(math.isfinite(v) for point in values for v in point):
        raise ValueError("Native preview requires finite head controls for framing")
    low, high = [min(v[i] for v in values) for i in range(3)], [
        max(v[i] for v in values) for i in range(3)]
    if high[0] <= low[0] or high[2] <= low[2]:
        raise ValueError("Native face controls have degenerate framing bounds")
    half_height = max((high[2]-low[2])*1.35/2, (high[0]-low[0])*1.15*height/width/2)
    return {"center_cm": [(a+b)/2 for a, b in zip(low, high)],
            "distance_cm": distance, "vertical_span_cm": half_height*2,
            "fov_degrees": math.degrees(2*math.atan(half_height*width/height/distance)),
            "basis": "SceneCapture horizontal FOV; native face-control bounds and output aspect"}


def main():
    import unreal

    request = json.loads(Path(os.environ["AILIAS_FRESH_VIDEO_REQUEST"]).read_text())
    receipt = Path(request["receipt"])
    result = {"status": "starting", "character": request["character"],
              "assembled": False, "likeness_verified": False, "cloud_requested": False}
    owned, character, callback = [], None, None
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    def publish():
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(receipt)

    def finish(error=None):
        nonlocal callback
        if error is not None:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
        failures = []
        if callback is not None:
            try:
                unreal.unregister_slate_post_tick_callback(callback)
            except Exception as cleanup:
                failures.append(type(cleanup).__name__)
            callback = None
        for actor in reversed(owned):
            try:
                actors.destroy_actor(actor)
            except Exception as cleanup:
                failures.append(type(cleanup).__name__)
        if character is not None:
            try:
                subsystem.remove_object_to_edit(character)
            except Exception as cleanup:
                failures.append(type(cleanup).__name__)
        if failures:
            result.update(status="failed", cleanup_errors=failures)
        try:
            publish()
        finally:
            unreal.EditorPythonScripting.set_keep_python_script_alive(False)

    try:
        if request["operation"] != "preview":
            raise ValueError("Unsupported editor preview operation")
        output = Path(request["output"])
        if output.exists():
            raise ValueError("Refusing to overwrite an existing preview")
        levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if not levels.new_level(request["destination"] + "/NativePreview"):
            raise RuntimeError("Cannot create isolated native preview map")
        character = unreal.load_asset(request["character"])
        if not isinstance(character, unreal.MetaHumanCharacter) or not subsystem.try_add_object_to_edit(character):
            raise RuntimeError("Cannot open newly saved native Character for preview")
        subject = subsystem.spawn_meta_human_actor(character, keep_transient=True)
        if subject is None:
            raise RuntimeError("Native Character did not produce an editor actor")
        owned.append(subject)
        subsystem.on_edit_preview_collection(character)
        subsystem.assemble_for_preview(character=character)
        preview_actors = [subject, *subject.get_all_child_actors(include_descendants=True)]
        groom_components = [component for actor in preview_actors
                            for component in actor.get_components_by_class(unreal.GroomComponent)]
        result["groom_components"] = [{
            "component": component.get_name(),
            "groom_asset": component.get_editor_property("groom_asset").get_path_name()
            if component.get_editor_property("groom_asset") is not None else None,
        } for component in groom_components]
        if not any(item["groom_asset"] for item in result["groom_components"]):
            raise RuntimeError("The newly assembled editor preview contains no actual groom assets")
        meshes = subject.get_components_by_class(unreal.SkeletalMeshComponent)
        faces = [component for component in meshes if "face" in component.get_name().lower()]
        if len(faces) != 1:
            raise RuntimeError("Native preview actor must expose exactly one face component")
        framing = face_framing(subsystem.get_face_landmarks(character))
        result["framing"] = framing
        center = unreal.Vector(*framing["center_cm"])
        lighting = request["lighting"]
        for side, strength in ((-1, 1), (1, lighting["fill_ratio"])):
            location = center + unreal.Vector(side*60, 80, 55)
            light = actors.spawn_actor_from_class(unreal.RectLight, location)
            owned.append(light)
            light.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(location, center), False)
            light.light_component.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
            light.light_component.set_editor_property("intensity", lighting["key_lumens"]*strength)
            light.light_component.set_editor_property("source_width", lighting["light_size_cm"])
            light.light_component.set_editor_property("source_height", lighting["light_size_cm"])
            light.light_component.set_editor_property("use_temperature", True)
            light.light_component.set_editor_property("temperature", lighting["temperature_kelvin"])
        capture_actor = actors.spawn_actor_from_class(unreal.SceneCapture2D, center + unreal.Vector(0, 75, 0))
        owned.append(capture_actor)
        capture_actor.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(
            capture_actor.get_actor_location(), center), False)
        capture = capture_actor.get_editor_property("capture_component2d")
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        target = unreal.RenderingLibrary.create_render_target2d(
            world, 540, 960, unreal.TextureRenderTargetFormat.RTF_RGBA8,
            unreal.LinearColor(.035, .035, .035, 1))
        capture.texture_target = target
        capture.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
        capture.fov_angle = framing["fov_degrees"]
        capture.capture_every_frame = True
        capture.capture_on_movement = False
        capture.post_process_blend_weight = 1
        settings = capture.post_process_settings
        settings.override_auto_exposure_method = True
        settings.auto_exposure_method = unreal.AutoExposureMethod.AEM_MANUAL
        settings.override_auto_exposure_bias = True
        settings.auto_exposure_bias = lighting["exposure_compensation_ev"]
        settings.override_auto_exposure_apply_physical_camera_exposure = True
        settings.auto_exposure_apply_physical_camera_exposure = False
        capture.post_process_settings = settings
        started = time.monotonic()
        captured_at = None
        result.update(status="warming_native_editor_preview", face_component=faces[0].get_name(),
                      render_size=[540, 960], output=str(output))
        publish()

        def tick(delta_seconds):
            nonlocal captured_at
            try:
                elapsed = time.monotonic()-started
                if elapsed < request["warmup_seconds"]:
                    return
                if captured_at is None:
                    capture.capture_scene()
                    captured_at = elapsed
                    return
                if elapsed-captured_at < 1:
                    return
                unreal.RenderingLibrary.export_render_target(
                    world, target, str(output.parent), output.name)
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError("Native scene capture did not write the requested PNG")
                result.update(status="native_editor_preview_rendered", seconds=round(elapsed, 3))
                finish()
            except Exception as error:
                finish(error)

        unreal.EditorPythonScripting.set_keep_python_script_alive(True)
        callback = unreal.register_slate_post_tick_callback(tick)
    except Exception as error:
        finish(error)
        raise


if __name__ == "__main__":
    main()
