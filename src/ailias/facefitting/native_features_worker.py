"""Edit a duplicated native Character; this worker runs in Unreal's isolated Python."""

import json
import math
import os
import re
import time
from pathlib import Path


GROOM_SLOTS = {"Hair", "Eyebrows", "Beard", "Mustache", "Eyelashes", "Peachfuzz"}
EYELASH_TYPES = {
    "none", "short_sparse", "short_fine", "short_thin",
    "long_slight_curl", "long_curl", "long_thick_curl",
}
EYELASH_FLOATS = {"melanin", "redness", "roughness", "salt_and_pepper", "lightness"}


def native_asset(value):
    if not isinstance(value, str) or not re.fullmatch(
            r"/(?:Game|MetaHumanCharacter)/[A-Za-z0-9_/-]+(?:\.[A-Za-z0-9_-]+)?", value):
        raise ValueError("Feature assets must be explicit Game or MetaHumanCharacter paths")
    if "//" in value or value.endswith("/"):
        raise ValueError("Feature asset path contains an empty component")
    if "." in value and value.rsplit(".", 1)[1] != value.split(".", 1)[0].rsplit("/", 1)[1]:
        raise ValueError("Feature object name must match its package")
    return value


def number(value, lower, upper):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not lower <= value <= upper):
        raise ValueError(f"Feature value must be finite and between {lower} and {upper}")
    return float(value)


def vector(value, lower, upper):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Feature vectors require three numbers")
    return [number(item, lower, upper) for item in value]


def validate_profile(profile):
    if not isinstance(profile, dict) or type(profile.get("version")) is not int or profile["version"] != 1:
        raise ValueError("Native feature profile version must be 1")
    allowed = {"version", "grooms", "eyes", "eyelashes", "face_basecolor", "sculpt",
               "sources", "assembly"}
    if profile.keys() - allowed:
        raise ValueError("Unknown native feature profile fields")
    grooms = profile.get("grooms", {})
    if not isinstance(grooms, dict) or grooms.keys() - GROOM_SLOTS:
        raise ValueError("Unsupported native groom slot")
    for settings in grooms.values():
        if not isinstance(settings, dict) or settings.keys() - {"asset", "parameters"}:
            raise ValueError("Groom requires an asset and optional float parameters")
        native_asset(settings.get("asset"))
        parameters = settings.get("parameters", {})
        if not isinstance(parameters, dict) or parameters.keys() - {"Melanin", "Redness", "Roughness"}:
            raise ValueError("Unsupported groom float parameter")
        for value in parameters.values():
            number(value, 0, 1)
    if "eyes" in profile:
        eyes = profile["eyes"]
        if (not isinstance(eyes, dict) or not eyes
                or eyes.keys() - {"global_tint_linear", "global_saturation"}):
            raise ValueError("Unsupported iris properties")
        if "global_tint_linear" in eyes:
            vector(eyes["global_tint_linear"], 0, 1)
        if "global_saturation" in eyes:
            number(eyes["global_saturation"], 0, 4)
    if "face_basecolor" in profile:
        native_asset(profile["face_basecolor"])
    if "eyelashes" in profile:
        lashes = profile["eyelashes"]
        if (not isinstance(lashes, dict) or not lashes
                or lashes.keys() - (EYELASH_FLOATS | {"type", "dye_color_linear", "enable_grooms"})):
            raise ValueError("Unsupported eyelash properties")
        if "type" in lashes and (not isinstance(lashes["type"], str) or lashes["type"] not in EYELASH_TYPES):
            raise ValueError("Unsupported native eyelash type")
        for key in EYELASH_FLOATS & lashes.keys():
            number(lashes[key], 0, 1)
        if "dye_color_linear" in lashes:
            vector(lashes["dye_color_linear"], 0, 1)
        if "enable_grooms" in lashes and type(lashes["enable_grooms"]) is not bool:
            raise ValueError("Eyelash enable_grooms must be a boolean")
        if "Eyelashes" in grooms and (
                lashes.get("type") == "none" or lashes.get("enable_grooms") is False):
            raise ValueError("An eyelash groom cannot be selected while explicitly disabled")
    if "assembly" in profile:
        assembly = profile["assembly"]
        if (not isinstance(assembly, dict) or not assembly
                or assembly.keys() - {"pipeline", "quality"}):
            raise ValueError("Unsupported native assembly settings")
        pipeline, quality = assembly.get("pipeline", "optimized"), assembly.get("quality", "medium")
        if not isinstance(pipeline, str) or pipeline not in {"optimized", "cinematic"}:
            raise ValueError("Assembly pipeline must be optimized or cinematic")
        if not isinstance(quality, str) or quality not in {"high", "medium", "low"}:
            raise ValueError("Assembly quality must be high, medium or low")
    if "sculpt" in profile:
        sculpt = profile["sculpt"]
        if not isinstance(sculpt, dict) or set(sculpt) != {
                "reference_landmarks", "deltas", "max_displacement_cm"}:
            raise ValueError("Sculpt requires an exact reference, deltas and displacement limit")
        limit = number(sculpt["max_displacement_cm"], 0.000001, 2)
        if not isinstance(sculpt["reference_landmarks"], list) or not sculpt["reference_landmarks"]:
            raise ValueError("Sculpt requires reference native landmarks")
        for point in sculpt["reference_landmarks"]:
            vector(point, -10000, 10000)
        if not isinstance(sculpt["deltas"], list) or not sculpt["deltas"]:
            raise ValueError("Sculpt requires nonempty landmark deltas")
        indices = set()
        for change in sculpt["deltas"]:
            if not isinstance(change, dict) or set(change) != {"index", "delta_cm"}:
                raise ValueError("Each sculpt delta requires an index and delta_cm")
            index = change["index"]
            if (isinstance(index, bool) or not isinstance(index, int)
                    or not 0 <= index < len(sculpt["reference_landmarks"]) or index in indices):
                raise ValueError("Sculpt landmark indices must be unique and in range")
            indices.add(index)
            delta = vector(change["delta_cm"], -limit, limit)
            if math.sqrt(sum(value*value for value in delta)) > limit + 1e-9:
                raise ValueError("Sculpt displacement exceeds its limit")
    if not any((grooms, profile.get("eyes"), profile.get("eyelashes"),
                profile.get("face_basecolor"), profile.get("sculpt"))):
        raise ValueError("Native feature profile does not request any edits")
    if not isinstance(profile.get("sources", []), list):
        raise ValueError("Profile sources must be hash references")
    for source in profile.get("sources", []):
        if (not isinstance(source, dict) or not isinstance(source.get("path"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", str(source.get("sha256", "")))):
            raise ValueError("Profile source requires path and SHA256")
    return profile


def landmarks(subsystem, character):
    return [[float(v.x), float(v.y), float(v.z)] for v in subsystem.get_face_landmarks(character)]


def maximum_distance(first, second):
    if not first or len(first) != len(second):
        raise ValueError("Native landmark correspondence changed")
    distances = [math.dist(a, b) for a, b in zip(first, second)]
    if not all(math.isfinite(value) for value in distances):
        raise ValueError("Native landmark evidence is nonfinite")
    return max(distances)


def selection_keys(collection):
    return [
        {"slot": str(item.selection.slot_name), "key": item.selection.selected_item.export_text()}
        for item in collection.default_instance.get_slot_selection_data()
    ]


def snapshot(character, subsystem, groom_keys, unreal):
    values = {"landmarks": landmarks(subsystem, character),
              "selections": selection_keys(character.internal_collection),
              "groom_parameters": {}}
    instance = subsystem.get_preview_collection(character).default_instance
    for slot, key in groom_keys.items():
        parameters = instance.get_instance_parameters(
            item_path=unreal.MetaHumanPaletteItemPath(item_key=key))
        values["groom_parameters"][slot] = {
            str(parameter.name): float(parameter.get_float())
            for parameter in parameters
            if str(parameter.name) in {"Melanin", "Redness", "Roughness"}
        }
    eyes = character.get_editor_property("eyes_settings")
    values["eyes"] = {}
    for name in ("eye_left", "eye_right"):
        iris = getattr(eyes, name).iris
        tint = iris.global_tint
        values["eyes"][name] = {
            "global_tint_linear": [float(tint.r), float(tint.g), float(tint.b)],
            "global_saturation": float(iris.global_saturation),
        }
    overrides = character.get_editor_property("skin_settings").texture_material_overrides
    texture = overrides.texture_overrides.face.get(unreal.FaceTextureType.BASECOLOR)
    values["face_basecolor"] = (
        texture.get_path_name() if overrides.enable_texture_overrides and texture else None)
    lashes = character.get_editor_property("head_model_settings").eyelashes
    color = lashes.dye_color
    values["eyelashes"] = {
        "type": next(name for name in sorted(EYELASH_TYPES)
                     if lashes.type == getattr(unreal.MetaHumanCharacterEyelashesType, name.upper())),
        "enable_grooms": bool(lashes.enable_grooms),
        "dye_color_linear": [float(color.r), float(color.g), float(color.b)],
        **{name: float(getattr(lashes, name)) for name in EYELASH_FLOATS},
    }
    return values


def verify_properties(wanted, actual, label):
    for key, target in wanted.items():
        value = actual.get(key)
        if isinstance(target, (str, bool)):
            matches = type(value) is type(target) and value == target
        else:
            target = target if isinstance(target, list) else [target]
            value = value if isinstance(value, list) else [value]
            matches = len(value) == len(target) and all(
                type(item) in (float, int) and math.isfinite(item)
                and abs(item-goal) <= 1e-5 for item, goal in zip(value, target))
        if not matches:
            raise RuntimeError(f"Native {label} properties did not persist")


def verify_profile(profile, values, expected=None):
    if "eyes" in profile:
        for name in ("eye_left", "eye_right"):
            verify_properties(profile["eyes"], values.get("eyes", {}).get(name, {}), "iris")
    if "eyelashes" in profile:
        verify_properties(profile["eyelashes"], values.get("eyelashes", {}), "eyelash")
    for slot, groom in profile.get("grooms", {}).items():
        if sum(item["slot"] == slot for item in values["selections"]) != 1:
            raise RuntimeError("Native groom slot does not have exactly one selection")
        actual = values["groom_parameters"].get(slot, {})
        for name, wanted in groom.get("parameters", {}).items():
            if name not in actual or abs(actual[name]-wanted) > 1e-5:
                raise RuntimeError("Native groom parameters did not persist")
    if "face_basecolor" in profile:
        if values["face_basecolor"] != profile["face_basecolor"]:
            raise RuntimeError("Native face basecolour override did not persist")
    if expected is not None:
        if maximum_distance(values["landmarks"], expected["landmarks"]) > 1e-4:
            raise RuntimeError("Native sculpt state changed after reopening")
        if sorted(values["selections"], key=str) != sorted(expected["selections"], key=str):
            raise RuntimeError("Native wardrobe selections changed after reopening")


def inspect_runtime(blueprint_path, required_grooms, unreal, publish=None):
    blueprint = unreal.load_asset(blueprint_path)
    if not isinstance(blueprint, unreal.Blueprint):
        raise RuntimeError("Runtime assembly did not produce a Blueprint")
    actor_class = blueprint.generated_class()
    if not actor_class or not isinstance(unreal.get_default_object(actor_class), unreal.Actor):
        raise RuntimeError("Runtime Blueprint does not generate an Actor")
    status = str(blueprint.get_editor_property("status"))
    if "UP_TO_DATE" not in status:
        raise RuntimeError("Runtime Blueprint is not compiled")
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actors.spawn_actor_from_class(actor_class, unreal.Vector())
    if actor is None:
        raise RuntimeError("Runtime Blueprint could not be instantiated")
    try:
        faces = [component for component in actor.get_components_by_class(unreal.SkeletalMeshComponent)
                 if component.get_name().lower() == "face"]
        if len(faces) != 1:
            raise RuntimeError("Runtime actor must have exactly one Face component")
        face = faces[0]
        mesh = face.get_skeletal_mesh_asset()
        if (mesh is None or face.get_bone_index("head") < 0
                or not mesh.get_editor_property("post_process_anim_blueprint")):
            raise RuntimeError("Runtime face is missing its rig or RigLogic post-process")
        grooms = []
        for component in actor.get_components_by_class(unreal.GroomComponent):
            groom = component.get_editor_property("groom_asset")
            binding = component.get_editor_property("binding_asset")
            if groom:
                grooms.append({"component": component.get_name(), "asset": groom.get_path_name(),
                               "binding": binding.get_path_name() if binding else None})
        evidence = {"blueprint": blueprint_path, "face_mesh": mesh.get_path_name(),
                    "head_bone_index": face.get_bone_index("head"), "grooms": grooms,
                    "blueprint_compile_status": status}
        if publish is not None:
            publish(evidence)
        for slot in required_grooms:
            matches = [item for item in grooms if item["component"].lower() == slot.lower()]
            if len(matches) != 1 or not matches[0]["binding"]:
                raise RuntimeError("Runtime actor is missing a requested bound groom: " + slot)
        return evidence
    finally:
        actors.destroy_actor(actor)


def assemble_runtime(character, subsystem, request, result, unreal, publish):
    result["stage"] = "check_native_build_prerequisites"
    publish(result)
    for property_name, stage_name in (
            ("has_face_dna_blendshapes", "request_native_full_face_rig"),
            ("has_high_resolution_textures", "request_native_texture_sources")):
        if character.get_editor_property(property_name):
            continue
        if not request.get("cloud_services", False):
            raise RuntimeError("Assembly needs missing native rig/textures; enable explicit cloud services")
        result.update(stage=stage_name, cloud_requested=True)
        publish(result)
        if property_name == "has_face_dna_blendshapes":
            parameters = unreal.MetaHumanCharacterAutoRiggingRequestParams()
            parameters.rig_type = unreal.MetaHumanRigType.JOINTS_AND_BLEND_SHAPES
            result["rig_requested"] = True
            operation = subsystem.request_auto_rigging
        else:
            parameters = unreal.MetaHumanCharacterTextureRequestParams()
            operation = subsystem.request_texture_sources
        parameters.blocking = True
        parameters.report_progress = False
        operation(character, parameters)
        if not character.get_editor_property(property_name):
            raise RuntimeError("Native rig/texture request did not satisfy assembly prerequisites")
    if not subsystem.can_build_meta_human(character, False):
        raise RuntimeError("Native Character is not ready for assembly")
    destination = request["destination"] + "/Runtime"
    if unreal.EditorAssetLibrary.list_assets(destination, recursive=True, include_folder=False):
        raise RuntimeError("Runtime destination is occupied; never overwrite it")
    parameters = unreal.MetaHumanCharacterEditorBuildParameters()
    assembly = request["profile"].get("assembly", {})
    pipeline, quality = assembly.get("pipeline", "optimized"), assembly.get("quality", "medium")
    parameters.pipeline_type = getattr(unreal.MetaHumanDefaultPipelineType, pipeline.upper())
    parameters.pipeline_quality = getattr(unreal.MetaHumanQualityLevel, quality.upper())
    result["assembly_settings"] = {
        "pipeline": pipeline, "quality": quality, "quality_applies": pipeline == "optimized",
    }
    parameters.absolute_build_path = destination
    parameters.common_folder_path = destination + "/Common"
    parameters.name_override = "FeatureDraft"
    result["stage"] = "assemble_native_runtime"
    publish(result)
    try:
        subsystem.build_meta_human(character, parameters)
    except RuntimeError as error:
        message = str(error)
        paths = re.findall(r"(/Game/\S+) :", message)
        permitted = {destination + "/Common/Face/Face_ControlBoard_CtrlRig",
                     destination + "/Common/Face/CR_MetaHuman_HeadMovement_IK_Proc"}
        if ("Cannot break link" not in message or not paths or set(paths) - permitted
                or any("Cannot break link" not in line for line in message.splitlines()
                       if "RuntimeError:" in line)):
            raise
        result["control_rig_upgrade_messages"] = message
    result["stage"] = "compile_runtime_blueprints"
    publish(result)
    assets = list(unreal.EditorAssetLibrary.list_assets(
        destination, recursive=True, include_folder=False))
    for path in assets:
        asset = unreal.load_asset(path)
        if not isinstance(asset, unreal.Blueprint):
            continue
        status = str(asset.get_editor_property("status"))
        if "UP_TO_DATE" not in status:
            unreal.BlueprintEditorLibrary.compile_blueprint(asset)
            status = str(asset.get_editor_property("status"))
        if "UP_TO_DATE" not in status:
            raise RuntimeError("Assembled Blueprint or Control Rig did not compile: " + path)
    blueprint = destination + "/FeatureDraft/BP_FeatureDraft.BP_FeatureDraft"
    result.update(stage="save_runtime_packages", blueprint=blueprint, runtime_assets=assets)
    publish(result)
    if not unreal.EditorAssetLibrary.save_directory(
            destination, only_if_is_dirty=False, recursive=True):
        raise RuntimeError("Runtime assembly packages could not be saved")

def run(request, unreal, publish):
    started = time.perf_counter()
    profile = validate_profile(request["profile"])
    if not unreal.SystemLibrary.get_engine_version().startswith("5.8."):
        raise ValueError("Native feature adapter targets UE 5.8")
    library = unreal.EditorAssetLibrary
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    result = {"status": "running", "stage": "load_character",
              "source": request["source"], "character": request["character"],
              "cloud_requested": False, "rig_requested": False,
              "likeness_verified": False, "visual_review_required": True}
    writer = publish
    phase = [None, started]

    def publish(value):
        now = time.perf_counter()
        if phase[0] is not None:
            timings = result.setdefault("stage_seconds", {})
            timings[phase[0]] = timings.get(phase[0], 0) + now-phase[1]
        phase[:] = [value["stage"], now]
        result["worker_seconds"] = now-started
        writer(value)

    def stage(name):
        result["stage"] = name
        publish(result)

    def save_character():
        stage("save_candidate_character")
        if not library.save_loaded_asset(character, only_if_is_dirty=False):
            raise RuntimeError("Candidate Character package could not be saved")

    stage("load_character")
    source = unreal.load_asset(request["source"])
    if not isinstance(source, unreal.MetaHumanCharacter):
        raise ValueError("Feature source must be an existing native Character")
    if subsystem.is_object_added_for_editing(source):
        raise RuntimeError("Source Character is already being edited")
    assets = {}
    for slot, settings in profile.get("grooms", {}).items():
        item = unreal.load_asset(settings["asset"])
        if not isinstance(item, unreal.MetaHumanWardrobeItem):
            raise ValueError("Groom must be a compatible native Wardrobe Item")
        assets[slot] = item
    if "face_basecolor" in profile:
        texture = unreal.load_asset(profile["face_basecolor"])
        if not isinstance(texture, unreal.Texture2D):
            raise ValueError("Face basecolour must be a native Texture2D")
    if request["operation"] == "apply":
        if library.list_assets(request["destination"], recursive=True, include_folder=False):
            raise ValueError("Feature destination is occupied; never overwrite it")
        stage("duplicate_existing_character")
        character = library.duplicate_asset(request["source"], request["character"])
    elif request["operation"] in {"verify", "build"}:
        character = unreal.load_asset(request["character"])
    else:
        raise ValueError("Unsupported native feature operation")
    if not isinstance(character, unreal.MetaHumanCharacter):
        raise RuntimeError("Candidate Character could not be loaded")
    if subsystem.is_object_added_for_editing(character):
        raise RuntimeError("Candidate Character is already being edited")
    collection = character.internal_collection
    instance = collection.default_instance
    if (not collection.get_path_name().startswith(character.get_path_name() + ":")
            or instance.get_meta_human_collection() != collection):
        raise RuntimeError("Candidate collection is not isolated from the original Character")
    stage("register_character_for_editing")
    if not subsystem.try_add_object_to_edit(character):
        raise RuntimeError("Candidate Character could not enter editing mode")
    try:
        # UE 5.8 edits a separate preview collection, not the asset's stored collection.
        collection = subsystem.get_preview_collection(character)
        if collection is None:
            raise RuntimeError("Candidate preview collection is unavailable")
        instance = collection.default_instance
        before = landmarks(subsystem, character)
        result["landmarks_before"] = before
        groom_keys = {}
        if request["operation"] == "apply":
            sculpt = profile.get("sculpt")
            if sculpt:
                if maximum_distance(before, sculpt["reference_landmarks"]) > 1e-4:
                    raise ValueError("Sculpt reference differs from the current native Character")
                stage("apply_bounded_native_sculpt")
                subsystem.translate_face_landmarks(
                    character, [change["index"] for change in sculpt["deltas"]],
                    [unreal.Vector(*change["delta_cm"]) for change in sculpt["deltas"]])
                subsystem.commit_face_state(character)
            if "eyes" in profile:
                stage("apply_iris_properties")
                settings = character.get_editor_property("eyes_settings")
                for name in ("eye_left", "eye_right"):
                    eye = getattr(settings, name)
                    iris = eye.iris
                    for key, value in profile["eyes"].items():
                        if key == "global_tint_linear":
                            iris.global_tint = unreal.LinearColor(*value, 1)
                        else:
                            iris.global_saturation = value
                    eye.iris = iris
                    setattr(settings, name, eye)
                subsystem.commit_eyes_settings(character, settings)
            if "eyelashes" in profile:
                stage("apply_eyelash_properties")
                settings = character.get_editor_property("head_model_settings")
                lashes = settings.eyelashes
                for key, value in profile["eyelashes"].items():
                    if key == "type":
                        lashes.type = getattr(unreal.MetaHumanCharacterEyelashesType, value.upper())
                    elif key == "dye_color_linear":
                        lashes.dye_color = unreal.LinearColor(*value, 1)
                    else:
                        setattr(lashes, key, value)
                settings.eyelashes = lashes
                subsystem.commit_head_model_settings(character, settings)
            if "face_basecolor" in profile:
                stage("apply_native_uv_basecolour")
                settings = character.get_editor_property("skin_settings")
                overrides = settings.texture_material_overrides
                textures = overrides.texture_overrides
                face = dict(textures.face)
                face[unreal.FaceTextureType.BASECOLOR] = texture
                textures.face = face
                overrides.texture_overrides = textures
                overrides.enable_texture_overrides = True
                settings.texture_material_overrides = overrides
                subsystem.commit_skin_settings(character, settings)
            slots = set(map(str, collection.get_slot_names()))
            for slot, item in assets.items():
                if slot not in slots:
                    raise ValueError("Native Character does not support requested groom slot")
                key = collection.try_add_item_from_wardrobe_item(slot, item)
                if key is None:
                    raise RuntimeError("Native wardrobe rejected the groom")
                instance.set_single_slot_selection(slot, key)
                groom_keys[slot] = key
            if groom_keys:
                subsystem.on_edit_preview_collection(character)
                stage("assemble_groom_preview")
                subsystem.assemble_for_preview(character=character)
                for slot, key in groom_keys.items():
                    parameters = {
                        str(parameter.name): parameter for parameter in instance.get_instance_parameters(
                            item_path=unreal.MetaHumanPaletteItemPath(item_key=key))}
                    result.setdefault("available_groom_parameters", {})[slot] = sorted(parameters)
                    publish(result)
                    for name, value in profile["grooms"][slot].get("parameters", {}).items():
                        if name not in parameters:
                            raise ValueError("Native groom does not expose requested parameter")
                        parameters[name].set_float(value=value)
                subsystem.on_edit_preview_collection(character)
            result["groom_keys"] = {slot: key.export_text() for slot, key in groom_keys.items()}
        else:
            for slot, settings in profile.get("grooms", {}).items():
                matching = list(collection.get_item_keys_for_wardrobe_item(assets[slot]))
                selections = {item["key"] for item in selection_keys(collection) if item["slot"] == slot}
                keys = [key for key in matching if key.export_text() in selections]
                if len(keys) != 1:
                    raise RuntimeError("Saved groom does not resolve to the requested Wardrobe Item")
                groom_keys[slot] = keys[0]
            if groom_keys:
                stage("reload_groom_preview")
                subsystem.assemble_for_preview(character=character)
        stage("verify_native_feature_state")
        values = snapshot(character, subsystem, groom_keys, unreal)
        verify_profile(profile, values, request.get("expected"))
        displacement = maximum_distance(before, values["landmarks"])
        if "sculpt" not in profile and displacement > 1e-4:
            raise RuntimeError("Appearance editing unexpectedly changed the native face")
        if "sculpt" in profile and request["operation"] == "apply":
            if displacement > profile["sculpt"]["max_displacement_cm"] + 1e-4:
                raise RuntimeError("Native sculpt exceeded its whole-face displacement budget")
        if request["operation"] == "apply":
            save_character()
        if request["operation"] == "build" or (
                request["operation"] == "apply" and request.get("assemble")):
            before_assembly = values
            assemble_runtime(character, subsystem, request, result, unreal, publish)
            stage("verify_built_feature_state")
            values = snapshot(character, subsystem, groom_keys, unreal)
            verify_profile(profile, values, before_assembly)
            save_character()
            stage("inspect_runtime_actor")
            result["runtime"] = inspect_runtime(
                result["blueprint"], profile.get("grooms", {}), unreal)
        if request["operation"] == "verify" and request.get("blueprint"):
            stage("verify_runtime_actor")
            result["runtime"] = inspect_runtime(
                request["blueprint"], profile.get("grooms", {}), unreal)
        status = {"apply": "features_saved", "build": "runtime_saved", "verify": "features_reloaded"}
        result.update(status=status[request["operation"]],
                      stage="awaiting_visual_review", values=values,
                      maximum_landmark_displacement_cm=displacement)
    finally:
        try:
            publish(result)
        finally:
            subsystem.remove_object_to_edit(character)
    publish(result)
    return result


def main():
    import unreal

    request = json.loads(Path(os.environ["AILIAS_FEATURE_REQUEST"]).read_text())
    receipt = Path(request["receipt"])

    def publish(value):
        receipt.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

    try:
        publish(run(request, unreal, publish))
    except Exception as error:
        result = json.loads(receipt.read_text()) if receipt.is_file() else {}
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      project_may_contain_partial_outputs=True)
        publish(result)
        raise


if __name__ == "__main__":
    main()
