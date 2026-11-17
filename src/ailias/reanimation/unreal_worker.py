"""Solve local audio and stage/render an existing rigged MetaHuman; never generate or repair identity geometry."""

import json
import os
from pathlib import Path

EXECUTOR = None
QUEUE = None
REQUEST = None
RESULT = None


def publish(request, result):
    Path(request["receipt"]).write_text(json.dumps(result, indent=2), encoding="utf-8")


def solve(request, unreal):
    destination = request["destination"]
    if unreal.EditorAssetLibrary.list_assets(destination, recursive=True, include_folder=False):
        raise ValueError("Reel destination is occupied; create a fresh job")
    task = unreal.AssetImportTask()
    task.filename = request["audio"]
    task.destination_path = destination
    task.destination_name = "Narration"
    task.automated = True
    task.save = True
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    tools.import_asset_tasks([task])
    waves = [unreal.load_asset(path) for path in task.imported_object_paths]
    waves = [asset for asset in waves if isinstance(asset, unreal.SoundWave)]
    if len(waves) != 1:
        raise ValueError("Narration did not import as exactly one SoundWave")
    performance = tools.create_asset("Performance", destination, unreal.MetaHumanPerformance,
                                     unreal.MetaHumanPerformanceFactoryNew())
    performance.set_editor_property("input_type", unreal.DataInputType.AUDIO)
    performance.set_editor_property("audio", waves[0])
    performance.set_editor_property("realtime_audio", False)
    performance.set_editor_property("generate_blinks", True)
    performance.set_editor_property("downmix_channels", False)
    performance.set_editor_property("audio_driven_animation_output_controls",
                                    unreal.AudioDrivenAnimationOutputControls.FULL_FACE)
    performance.set_editor_property("head_movement_mode", unreal.PerformanceHeadMovementMode.CONTROL_RIG)
    overrides = unreal.AudioDrivenAnimationSolveOverrides()
    overrides.mood = unreal.AudioDrivenAnimationMood.EXCITEMENT
    overrides.mood_intensity = request["policy"]["mood_intensity"]
    performance.set_editor_property("audio_driven_animation_solve_overrides", overrides)
    performance.set_blocking_processing(True)
    code = performance.start_pipeline()
    if code != unreal.StartPipelineErrorType.NONE:
        raise RuntimeError("Audio solver did not start successfully: " + str(code))
    count = performance.get_number_of_processed_frames()
    if count <= 0 or performance.is_processing():
        raise RuntimeError("Audio solver did not finish with actual processed frames")
    if not unreal.EditorAssetLibrary.save_loaded_asset(performance, only_if_is_dirty=False):
        raise RuntimeError("Audio performance package was not saved")
    return performance, {"performance": performance.get_path_name(), "processed_frames": count,
                         "sound": waves[0].get_path_name(), "mood": "EXCITEMENT",
                         "solver_device": "CPU: installed Speech2Face uses NNERuntimeORTCpu",
                         "head_movement": "CONTROL_RIG", "blinks": True}


def rig_class(path, unreal):
    blueprint = unreal.load_asset(path)
    if not isinstance(blueprint, unreal.Blueprint):
        raise ValueError("Input must be an assembled rigged MetaHuman Blueprint, not a Character draft/static mesh")
    actor_class = blueprint.generated_class()
    if not actor_class:
        raise ValueError("MetaHuman Blueprint has no compiled generated class")
    return blueprint, actor_class


def linear_colour(rgb, unreal, exposure_compensation_ev=0):
    def convert(value):
        return value/12.92 if value <= .04045 else ((value+.055)/1.055)**2.4
    return unreal.LinearColor(*(convert(value)*2**(-exposure_compensation_ev) for value in rgb), 1)


def normalize_subject_root(sequence, actor_class, unreal):
    subjects = [binding for binding in sequence.get_bindings()
                if (template := binding.get_object_template()) is not None
                and template.get_class() == actor_class]
    if len(subjects) != 1:
        raise ValueError("Export must contain exactly one bound MetaHuman actor")
    tracks = [track for track in subjects[0].get_tracks()
              if isinstance(track, unreal.MovieScene3DTransformTrack)]
    if len(tracks) != 1 or len(tracks[0].get_sections()) != 1:
        raise ValueError("Exported MetaHuman must have one static root transform section")
    section = tracks[0].get_sections()[0]
    channels = section.get_channels_by_type(unreal.MovieSceneScriptingDoubleChannel)
    if len(channels) != 9 or any(channel.get_num_keys() for channel in channels):
        raise ValueError("Refusing to replace an unexpected or animated MetaHuman root")
    # Native audio export adds a reference-pose offset/yaw unrelated to our origin-framed camera.
    for channel, value in zip(channels, (0, 0, 0, 0, 0, 0, 1, 1, 1)):
        channel.set_default(value)


def stage(request, performance, unreal):
    policy, destination = request["policy"], request["destination"]
    blueprint, actor_class = rig_class(request["blueprint"], unreal)
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not levels.new_level(destination + "/PortraitStage"):
        raise RuntimeError("Could not create an isolated portrait map")
    reference = actors.spawn_actor_from_class(actor_class, unreal.Vector())
    try:
        components = reference.get_components_by_class(unreal.SkeletalMeshComponent)
        faces = [component for component in components if component.get_name().lower() == "face"]
        if len(faces) != 1 or not faces[0].get_skeletal_mesh_asset():
            raise ValueError("Rigged MetaHuman must expose one Face skeletal component")
        face = faces[0]
        if face.get_bone_index("head") < 0:
            raise ValueError("MetaHuman face skeleton has no head bone for stable framing")
        center = face.get_socket_location("head") + unreal.Vector(0, 0, policy["head_center_offset_cm"])
        if not face.get_skeletal_mesh_asset().get_editor_property("post_process_anim_blueprint"):
            raise ValueError("Face has no RigLogic post-process animation blueprint")
    finally:
        actors.destroy_actor(reference)
    settings = unreal.MetaHumanPerformanceExportLevelSequenceSettings()
    settings.show_export_dialog = False
    settings.package_path = destination
    settings.asset_name = "PortraitReel"
    settings.target_meta_human_class = blueprint
    for name in ("export_video_track", "export_depth_track", "export_image_plane",
                 "export_identity", "export_camera", "export_transform_track"):
        settings.set_editor_property(name, False)
    settings.export_audio_track = True
    settings.export_control_rig_track = True
    settings.enable_meta_human_head_movement = True
    settings.enable_control_rig_head_movement = True
    settings.keep_frame_range = False
    sequence = unreal.MetaHumanPerformanceExportUtils.export_level_sequence(performance, settings)
    if not isinstance(sequence, unreal.LevelSequence):
        raise RuntimeError("Native audio performance did not export a playable MetaHuman sequence")
    normalize_subject_root(sequence, actor_class, unreal)
    rate = sequence.get_display_rate()
    # Export owns key timing. Converting the whole sequence by changing display rate would retime keys.
    frames = request["frame_count"]
    sequence.set_playback_start_seconds(0)
    sequence.set_playback_end_seconds(frames/policy["fps"])
    direction = {"+X": (1, 0, 0), "-X": (-1, 0, 0),
                 "+Y": (0, 1, 0), "-Y": (0, -1, 0)}[policy["front_axis"]]
    front = unreal.Vector(*direction)
    right = unreal.Vector(direction[1], -direction[0], 0)
    location = center + front*request["framing"]["camera_distance_cm"]
    camera = actors.spawn_actor_from_class(unreal.CineCameraActor, location)
    camera.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(location, center), False)
    component = camera.get_cine_camera_component()
    filmback = component.get_editor_property("filmback")
    filmback.sensor_width = request["framing"]["sensor_width_mm"]
    filmback.sensor_height = policy["sensor_height_mm"]
    component.set_editor_property("filmback", filmback)
    component.set_editor_property("current_focal_length", policy["focal_length_mm"])
    component.set_editor_property("current_aperture", 5.6)
    post_process = component.get_editor_property("post_process_settings")
    for name, value in {
        "auto_exposure_method": unreal.AutoExposureMethod.AEM_MANUAL,
        "auto_exposure_apply_physical_camera_exposure": False,
        "auto_exposure_bias": policy["exposure_compensation_ev"],
        "bloom_intensity": 0, "vignette_intensity": 0, "lens_flare_intensity": 0,
    }.items():
        post_process.set_editor_property("override_" + name, True)
        post_process.set_editor_property(name, value)
    component.set_editor_property("post_process_settings", post_process)
    component.set_editor_property("post_process_blend_weight", 1)
    focus = component.get_editor_property("focus_settings")
    focus.focus_method = unreal.CameraFocusMethod.DISABLE
    component.set_editor_property("focus_settings", focus)
    binding = sequence.add_possessable(camera)
    cuts = sequence.add_track(unreal.MovieSceneCameraCutTrack)
    cut = cuts.add_section()
    cut.set_range_seconds(0, frames/policy["fps"])
    cut.set_camera_binding_id(unreal.MovieSceneSequenceExtensions.get_binding_id(sequence, binding))
    background = actors.spawn_actor_from_class(unreal.StaticMeshActor, center-front*150)
    background.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Plane"))
    background.set_actor_rotation(unreal.MathLibrary.make_rot_from_z(front), False)
    background.set_actor_scale3d(unreal.Vector(20, 20, 20))
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    material = tools.create_asset("ChromaBackdrop", destination, unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    expression = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionConstant3Vector)
    expression.set_editor_property("constant", linear_colour(
        policy["background_srgb"], unreal, policy["exposure_compensation_ev"]))
    unreal.MaterialEditingLibrary.connect_material_property(expression, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    unreal.MaterialEditingLibrary.recompile_material(material)
    background.static_mesh_component.set_material(0, material)
    for label, side, strength in (("SoftKey", -1, 1), ("SoftFill", 1, policy["fill_ratio"])):
        light_location = center + front*100 + right*(side*80) + unreal.Vector(0, 0, 65)
        light = actors.spawn_actor_from_class(unreal.RectLight, light_location)
        light.set_actor_label(label)
        light.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(light_location, center), False)
        light.light_component.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
        light.light_component.set_intensity(policy["key_lumens"]*strength)
        light.light_component.set_editor_property("source_width", policy["light_size_cm"])
        light.light_component.set_editor_property("source_height", policy["light_size_cm"])
        light.light_component.set_editor_property("use_temperature", True)
        light.light_component.set_temperature(policy["temperature_kelvin"])
    unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False)
    if not (unreal.EditorAssetLibrary.save_loaded_asset(sequence, only_if_is_dirty=False)
            and levels.save_current_level()):
        raise RuntimeError("Portrait stage or sequence did not save")
    return sequence, {"sequence": sequence.get_path_name(), "map": destination + "/PortraitStage",
                      "material": material.get_path_name(),
                      "camera": {"location": [location.x, location.y, location.z],
                                 "target": [center.x, center.y, center.z], **request["framing"]},
                      "native_sequence_fps": rate.numerator/rate.denominator,
                      "subject_root": "Identity transform; native face/head Control Rig keys preserved",
                      "focus": "Depth of field disabled for sharp face and keyable edges",
                      "body_gestures": "Not synthesized; audio generates face and head movement only"}


def finished(executor, success):
    import unreal

    RESULT.update(status="frames_rendered" if success else "failed",
                  stage="verify_frames_and_encode" if success else "native_render_failed",
                  render_success=bool(success))
    publish(REQUEST, RESULT)
    # -ExecutePythonScript owns exit and must destroy its notification before quitting.
    unreal.EditorPythonScripting.set_keep_python_script_alive(False)


def render(request, sequence, result, unreal):
    global EXECUTOR, QUEUE, REQUEST, RESULT
    REQUEST, RESULT = request, result
    QUEUE = unreal.MoviePipelineQueue()
    job = QUEUE.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
    job.map = unreal.SoftObjectPath(result["map"])
    config = job.get_configuration()
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    policy = request["policy"]
    output.output_directory = unreal.DirectoryPath(request["frames"])
    output.file_name_format = "frame.{frame_number_rel}"
    output.zero_pad_frame_numbers = 6
    output.output_resolution = unreal.IntPoint(policy["width"], policy["height"])
    output.use_custom_frame_rate = True
    output.output_frame_rate = unreal.FrameRate(policy["fps"], 1)
    output.override_existing_output = False
    output.flush_disk_writes_per_shot = True
    config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    config.find_or_add_setting_by_class(unreal.MoviePipelineColorSetting).disable_tone_curve = True
    game = config.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    game.game_mode_override = unreal.MoviePipelineGameMode
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.temporal_sample_count = policy["temporal_samples"]
    aa.engine_warm_up_count = policy["warmup_frames"]
    aa.render_warm_up_count = 8
    aa.use_camera_cut_for_warm_up = False
    EXECUTOR = unreal.MoviePipelinePIEExecutor()
    EXECUTOR.on_executor_finished_delegate.add_callable_unique(finished)
    result.update(status="rendering", stage="gpu_movie_render_queue")
    publish(request, result)
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    EXECUTOR.execute(QUEUE)


def main():
    import unreal

    request = json.loads(Path(os.environ["AILIAS_REANIMATION_REQUEST"]).read_text())
    result = {"status": "running", "stage": "audio_solve", "cloud_requested": False,
              "engine": unreal.SystemLibrary.get_engine_version(),
              "render_device": "D3D12 GPU", "likeness_generation": False}
    publish(request, result)
    try:
        if not result["engine"].startswith("5.8."):
            raise ValueError("Reanimation adapter targets the verified UE 5.8 API")
        if request.get("blueprint"):
            rig_class(request["blueprint"], unreal)
        performance, evidence = solve(request, unreal)
        result.update(evidence)
        if not request.get("blueprint"):
            result.update(status="audio_solved_awaiting_rigged_metahuman", stage="rigged_input_required")
        else:
            result.update(stage="portrait_stage")
            publish(request, result)
            sequence, evidence = stage(request, performance, unreal)
            result.update(evidence)
            render(request, sequence, result, unreal)
            return
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      project_may_contain_partial_outputs=True)
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)
    publish(request, result)


if __name__ == "__main__":
    main()
