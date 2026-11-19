"""Verify native edits and immutable revision handling without loading Unreal or models."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ailias.metahuman.character_features import apply_character_features, character_package
from ailias.metahuman.character_features_worker import (
    assemble_runtime, bind_runtime_skin, inspect_runtime_skin, run, validate_profile, verify_profile,
)
from ailias.storage import Store


class NativeFeatureWorkerTests(unittest.TestCase):
    def fixture(self):
        source_path = "/Game/Original/Head.Head"
        candidate_path = "/Game/Ailias/Features/Ftest/Character.Character"

        class Key:
            def export_text(self):
                return "short-beard-key"

        class Parameter:
            name = "Melanin"
            value = .5

            def get_float(self):
                return self.value

            def set_float(self, *, value):
                self.value = value

        class Wardrobe:
            pass

        class Character:
            def __init__(self, path):
                self.path = path
                self.points = [SimpleNamespace(x=0, y=0, z=0), SimpleNamespace(x=1, y=1, z=1)]
                self.key, self.parameter = Key(), Parameter()
                self.selected = {}
                instance = SimpleNamespace(
                    get_meta_human_collection=lambda: self.internal_collection,
                    get_slot_selection_data=lambda: [
                        SimpleNamespace(selection=SimpleNamespace(slot_name=slot, selected_item=key))
                        for slot, key in self.selected.items()],
                    set_single_slot_selection=lambda slot, key: self.selected.update({slot: key}),
                    get_instance_parameters=lambda **kwargs: [self.parameter],
                )
                self.internal_collection = SimpleNamespace(
                    default_instance=instance,
                    get_path_name=lambda: path + ":Collection",
                    get_slot_names=lambda: ["Beard"],
                    try_add_item_from_wardrobe_item=lambda *args: self.key,
                    get_item_keys_for_wardrobe_item=lambda item: [self.key],
                )
                iris = SimpleNamespace(
                    global_tint=SimpleNamespace(r=1., g=1., b=1.), global_saturation=1.7,
                    primary_color_u=.5, primary_color_v=.5,
                    secondary_color_u=.5, secondary_color_v=.5,
                    color_blend=.5, color_blend_softness=.5)
                self.eyes = SimpleNamespace(
                    eye_left=SimpleNamespace(iris=copy.deepcopy(iris)),
                    eye_right=SimpleNamespace(iris=copy.deepcopy(iris)))
                self.skin = SimpleNamespace(
                    skin=SimpleNamespace(u=.5, v=.5, roughness=1.06, face_texture_index=0),
                    freckles=SimpleNamespace(mask="NONE", density=.5, strength=.2,
                                             saturation=.6, tone_shift=.65),
                    accents=SimpleNamespace(**{
                        region: SimpleNamespace(redness=.5, saturation=.5, lightness=.5)
                        for region in ("scalp", "forehead", "nose", "under_eye",
                                       "cheeks", "lips", "chin", "ears")}),
                    texture_material_overrides=SimpleNamespace(
                    enable_texture_overrides=False, texture_overrides=SimpleNamespace(face={}, body={})))
                self.preview_material = "TOPOLOGY"
                self.head = SimpleNamespace(eyelashes=SimpleNamespace(
                    type="NONE", dye_color=SimpleNamespace(r=1., g=1., b=1.),
                    melanin=.3, redness=.28, roughness=.25, salt_and_pepper=.2,
                    lightness=.5, enable_grooms=True))

            def get_path_name(self):
                return self.path

            def get_editor_property(self, name):
                return {"eyes_settings": self.eyes, "skin_settings": self.skin,
                        "preview_material_type": self.preview_material,
                        "head_model_settings": self.head}[name]

            def set_editor_property(self, name, value):
                if name != "preview_material_type":
                    raise ValueError("Unexpected fixture property")
                self.preview_material = value

        source, candidate = Character(source_path), Character(candidate_path)
        wardrobe = Wardrobe()
        subsystem, library = Mock(), Mock()
        subsystem.is_object_added_for_editing.return_value = False
        subsystem.try_add_object_to_edit.return_value = True
        subsystem.get_face_landmarks.side_effect = lambda character: character.points
        subsystem.get_preview_collection.side_effect = lambda character: character.internal_collection
        subsystem.commit_eyes_settings.side_effect = lambda character, settings: setattr(
            character, "eyes", settings)
        subsystem.commit_head_model_settings.side_effect = lambda character, settings: setattr(
            character, "head", settings)
        subsystem.commit_skin_settings.side_effect = lambda character, settings: setattr(
            character, "skin", settings)
        library.list_assets.return_value = []
        library.duplicate_asset.return_value = candidate
        library.save_loaded_asset.return_value = True
        unreal = SimpleNamespace(
            SystemLibrary=SimpleNamespace(get_engine_version=lambda: "5.8.4-fixture"),
            EditorAssetLibrary=library, MetaHumanCharacter=Character,
            MetaHumanCharacterEditorSubsystem=object,
            get_editor_subsystem=lambda cls: subsystem,
            MetaHumanWardrobeItem=Wardrobe, Texture2D=object,
            load_asset=lambda path: source if path == source_path else (
                candidate if path == candidate_path else wardrobe),
            MetaHumanPaletteItemPath=lambda **kwargs: kwargs,
            FaceTextureType=SimpleNamespace(BASECOLOR="basecolor"),
            MetaHumanCharacterSkinPreviewMaterial=SimpleNamespace(
                DEFAULT="TOPOLOGY", EDITABLE="SKIN", CLAY="CLAY"),
            MetaHumanCharacterFrecklesMask=SimpleNamespace(
                NONE="NONE", TYPE1="TYPE1", TYPE2="TYPE2", TYPE3="TYPE3"),
            MetaHumanCharacterEyelashesType=SimpleNamespace(**{
                name.upper(): name.upper() for name in (
                    "none", "short_sparse", "short_fine", "short_thin",
                    "long_slight_curl", "long_curl", "long_thick_curl")}),
            LinearColor=lambda r, g, b, a: SimpleNamespace(r=r, g=g, b=b, a=a),
            Vector=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
        )
        request = {
            "operation": "apply", "source": source_path, "character": candidate_path,
            "destination": "/Game/Ailias/Features/Ftest",
            "profile": {"version": 1, "grooms": {"Beard": {
                "asset": "/MetaHumanCharacter/Grooms/Short.Short", "parameters": {"Melanin": .95}}},
                "eyes": {"global_tint_linear": [.2, .1, .05]}},
        }
        return unreal, subsystem, source, candidate, request

    def test_feature_revision_applies_groom_and_eyes_without_changing_original_or_shape(self):
        unreal, subsystem, source, candidate, request = self.fixture()
        result = run(request, unreal, lambda value: None)
        self.assertEqual(result["status"], "features_saved")
        self.assertFalse(result["likeness_verified"])
        self.assertEqual(candidate.parameter.value, .95)
        self.assertEqual(source.parameter.value, .5)
        self.assertEqual(candidate.eyes.eye_left.iris.global_tint.r, .2)
        self.assertEqual(source.eyes.eye_left.iris.global_tint.r, 1)
        self.assertEqual(result["maximum_landmark_displacement_cm"], 0)
        subsystem.remove_object_to_edit.assert_called_once_with(candidate)
        subsystem.request_auto_rigging.assert_not_called()
        subsystem.commit_face_state.assert_not_called()
        self.assertEqual(subsystem.on_edit_preview_collection.call_count, 2)
        unreal.EditorAssetLibrary.save_loaded_asset.assert_called_once_with(
            candidate, only_if_is_dirty=False)

    def test_reopening_verifies_actual_saved_selection_and_parameter_values(self):
        unreal, _, _, candidate, request = self.fixture()
        result = run(request, unreal, lambda value: None)
        request.update(operation="verify", expected=result["values"])
        self.assertEqual(run(request, unreal, lambda value: None)["status"], "features_reloaded")
        candidate.parameter.value = .1
        with self.assertRaisesRegex(RuntimeError, "parameters did not persist"):
            run(request, unreal, lambda value: None)

    def test_explicit_lashes_persist_type_pigment_and_groom_enablement_without_editing_source(self):
        unreal, subsystem, source, candidate, request = self.fixture()
        request["profile"]["eyelashes"] = {
            "type": "short_sparse", "melanin": .95, "redness": 0.,
            "salt_and_pepper": 0., "dye_color_linear": [.8, .7, .6], "enable_grooms": True,
        }
        result = run(request, unreal, lambda value: None)
        self.assertEqual(candidate.head.eyelashes.type, "SHORT_SPARSE")
        self.assertEqual(source.head.eyelashes.type, "NONE")
        self.assertEqual(candidate.head.eyelashes.melanin, .95)
        subsystem.commit_head_model_settings.assert_called_once_with(candidate, candidate.head)
        request.update(operation="verify", expected=result["values"])
        self.assertEqual(run(request, unreal, lambda value: None)["status"], "features_reloaded")
        candidate.head.eyelashes.enable_grooms = False
        with self.assertRaisesRegex(RuntimeError, "eyelash properties did not persist"):
            run(request, unreal, lambda value: None)

    def test_native_skin_and_iris_palette_edits_save_and_reload_without_sculpting(self):
        unreal, subsystem, source, candidate, request = self.fixture()
        request["profile"]["skin"] = {"u": .4, "v": .7, "roughness": 1.1}
        request["profile"]["eyes"].update(primary_color_u=.2, primary_color_v=.3)
        result = run(request, unreal, lambda value: None)
        self.assertEqual(result["values"]["skin"],
                         {**request["profile"]["skin"], "face_texture_index": 0})
        self.assertEqual(source.skin.skin.u, .5)
        self.assertEqual(source.eyes.eye_left.iris.primary_color_u, .5)
        self.assertEqual(candidate.eyes.eye_left.iris.primary_color_u, .2)
        self.assertEqual(candidate.eyes.eye_right.iris.primary_color_v, .3)
        self.assertEqual(candidate.skin.skin.roughness, 1.1)
        self.assertEqual(source.preview_material, "TOPOLOGY")
        self.assertTrue(result["values"]["skin_preview_enabled"])
        subsystem.translate_face_landmarks.assert_not_called()
        subsystem.commit_skin_settings.assert_called_once_with(candidate, candidate.skin)
        request.update(operation="verify", expected=result["values"])
        self.assertEqual(run(request, unreal, lambda value: None)["status"], "features_reloaded")
        candidate.skin.skin.v = .1
        with self.assertRaisesRegex(RuntimeError, "skin properties did not persist"):
            run(request, unreal, lambda value: None)

    def test_persisted_skin_values_cannot_pass_when_the_viewport_still_hides_colour(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        request["profile"] = {"version": 1, "skin": {"u": .4}}
        result = run(request, unreal, lambda value: None)
        self.assertTrue(result["values"]["skin_preview_enabled"])
        subsystem.translate_face_landmarks.assert_not_called()
        request.update(operation="verify", expected=result["values"])
        candidate.preview_material = "TOPOLOGY"
        with self.assertRaisesRegex(RuntimeError, "topology/clay hides skin colour"):
            run(request, unreal, lambda value: None)

    def test_explicit_native_skin_removes_photo_override_and_detects_its_return_on_reload(self):
        unreal, subsystem, source, candidate, request = self.fixture()
        texture = SimpleNamespace(get_path_name=lambda: "/Game/Photo.Photo")
        for character in (source, candidate):
            overrides = character.skin.texture_material_overrides
            overrides.enable_texture_overrides = True
            overrides.texture_overrides.face = {"basecolor": texture, "normal": texture}
        request["profile"] = {
            "version": 1, "skin": {"u": .35, "v": .9}, "face_basecolor": None}
        result = run(request, unreal, lambda value: None)
        self.assertIsNone(result["values"]["face_basecolor"])
        self.assertEqual(source.skin.texture_material_overrides.texture_overrides.face["basecolor"], texture)
        self.assertEqual(candidate.skin.texture_material_overrides.texture_overrides.face, {"normal": texture})
        subsystem.request_auto_rigging.assert_not_called()
        request.update(operation="verify", expected=result["values"])
        self.assertEqual(run(request, unreal, lambda value: None)["status"], "features_reloaded")
        candidate.skin.texture_material_overrides.texture_overrides.face["basecolor"] = texture
        with self.assertRaisesRegex(RuntimeError, "basecolour override did not persist"):
            run(request, unreal, lambda value: None)

    def test_removing_a_photo_override_alone_is_a_valid_edit(self):
        unreal, _, _, _, request = self.fixture()
        request["profile"] = {"version": 1, "face_basecolor": None}
        self.assertEqual(run(request, unreal, lambda value: None)["status"], "features_saved")

    def test_native_reset_disables_an_empty_override_layer_but_preserves_body_overrides(self):
        for body in ({}, {"basecolor": object()}):
            with self.subTest(body=bool(body)):
                unreal, _, _, candidate, request = self.fixture()
                overrides = candidate.skin.texture_material_overrides
                overrides.enable_texture_overrides = True
                overrides.texture_overrides.face = {
                    "basecolor": SimpleNamespace(get_path_name=lambda: "/Game/Photo.Photo")}
                overrides.texture_overrides.body = body
                request["profile"] = {"version": 1, "face_basecolor": None}
                result = run(request, unreal, lambda value: None)
                self.assertIsNone(result["values"]["face_basecolor"])
                self.assertEqual(overrides.enable_texture_overrides, bool(body))
                self.assertEqual(overrides.texture_overrides.body, body)

    def test_complexion_and_lip_edits_persist_without_sculpting_or_changing_source(self):
        unreal, subsystem, source, candidate, request = self.fixture()
        request["profile"] = {
            "version": 1, "skin": {"face_texture_index": 7, "roughness": 1.12},
            "freckles": {"mask": "none", "strength": 0.},
            "accents": {"lips": {"redness": .4, "saturation": .38, "lightness": .42},
                        "cheeks": {"redness": .4}},
        }
        result = run(request, unreal, lambda value: None)
        self.assertEqual(candidate.skin.skin.face_texture_index, 7)
        self.assertEqual(source.skin.skin.face_texture_index, 0)
        self.assertEqual(candidate.skin.accents.lips.lightness, .42)
        self.assertEqual(candidate.skin.accents.nose.lightness, .5)
        self.assertEqual(source.skin.accents.lips.lightness, .5)
        self.assertEqual(candidate.parameter.value, .5)
        self.assertTrue(result["values"]["skin_preview_enabled"])
        subsystem.commit_skin_settings.assert_called_once_with(candidate, candidate.skin)
        subsystem.commit_head_model_settings.assert_not_called()
        subsystem.translate_face_landmarks.assert_not_called()
        subsystem.commit_face_state.assert_not_called()
        request.update(operation="verify", expected=result["values"])
        self.assertEqual(run(request, unreal, lambda value: None)["status"], "features_reloaded")
        candidate.skin.accents.lips.lightness = .5
        with self.assertRaisesRegex(RuntimeError, "skin accent properties did not persist"):
            run(request, unreal, lambda value: None)

    def test_freckles_or_accents_alone_enable_skin_and_detect_lost_edits_on_reload(self):
        for edit in ({"freckles": {"mask": "type2", "strength": .1}},
                     {"accents": {"forehead": {"redness": .4}}}):
            with self.subTest(edit=edit):
                unreal, _, _, candidate, request = self.fixture()
                request["profile"] = {"version": 1, **edit}
                result = run(request, unreal, lambda value: None)
                self.assertTrue(result["values"]["skin_preview_enabled"])
                request.update(operation="verify", expected=result["values"])
                candidate.preview_material = "TOPOLOGY"
                with self.assertRaisesRegex(RuntimeError, "topology/clay hides skin colour"):
                    run(request, unreal, lambda value: None)
                candidate.preview_material = "SKIN"
                candidate.skin.freckles.strength = .2
                candidate.skin.accents.forehead.redness = .5
                with self.assertRaisesRegex(RuntimeError, "freckles|skin accent"):
                    run(request, unreal, lambda value: None)

    def test_preview_edits_are_copied_back_into_the_saved_collection(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        preview = type(candidate)(candidate.path + ":Preview")
        subsystem.get_preview_collection.side_effect = lambda character: preview.internal_collection

        def copy_preview(character):
            character.selected = dict(preview.selected)
            character.parameter.value = preview.parameter.value

        subsystem.on_edit_preview_collection.side_effect = copy_preview
        result = run(request, unreal, lambda value: None)
        self.assertEqual(preview.parameter.value, .95)
        self.assertEqual(candidate.parameter.value, .95)
        self.assertEqual(result["values"]["selections"][0]["slot"], "Beard")

    def test_stale_sculpt_reference_never_saves_or_changes_the_face(self):
        unreal, subsystem, _, _, request = self.fixture()
        request["profile"]["sculpt"] = {
            "reference_landmarks": [[9, 0, 0], [1, 1, 1]],
            "max_displacement_cm": .5, "deltas": [{"index": 0, "delta_cm": [.1, 0, 0]}],
        }
        with self.assertRaisesRegex(ValueError, "reference differs"):
            run(request, unreal, lambda value: None)
        subsystem.translate_face_landmarks.assert_not_called()
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        subsystem.remove_object_to_edit.assert_called_once()

    def test_sculpt_edits_only_the_isolated_candidate_and_commits_native_state(self):
        unreal, subsystem, source, candidate, request = self.fixture()
        request["profile"]["sculpt"] = {
            "reference_landmarks": [[0, 0, 0], [1, 1, 1]],
            "max_displacement_cm": .5, "deltas": [{"index": 0, "delta_cm": [.1, 0, 0]}],
        }

        def translate(character, indices, deltas):
            for index, delta in zip(indices, deltas):
                point = character.points[index]
                point.x, point.y, point.z = point.x+delta.x, point.y+delta.y, point.z+delta.z

        subsystem.translate_face_landmarks.side_effect = translate
        result = run(request, unreal, lambda value: None)
        self.assertAlmostEqual(result["maximum_landmark_displacement_cm"], .1)
        self.assertAlmostEqual(candidate.points[0].x, .1)
        self.assertEqual(source.points[0].x, 0)
        subsystem.commit_face_state.assert_called_once_with(candidate)

    def test_occupied_destination_and_busy_source_are_not_taken_over(self):
        for busy in (False, True):
            with self.subTest(busy=busy):
                unreal, subsystem, _, _, request = self.fixture()
                subsystem.is_object_added_for_editing.return_value = busy
                unreal.EditorAssetLibrary.list_assets.return_value = ["/Game/Existing"]
                with self.assertRaisesRegex((ValueError, RuntimeError), "occupied|already being edited"):
                    run(request, unreal, lambda value: None)
                unreal.EditorAssetLibrary.duplicate_asset.assert_not_called()
                subsystem.remove_object_to_edit.assert_not_called()

    def test_preview_error_releases_only_candidate_and_never_saves(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        subsystem.assemble_for_preview.side_effect = RuntimeError("fixture failure")
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            run(request, unreal, lambda value: None)
        subsystem.remove_object_to_edit.assert_called_once_with(candidate)
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()

    def test_assembly_never_requests_missing_cloud_prerequisites_implicitly(self):
        unreal, subsystem, _, character, request = self.fixture()
        character.get_editor_property = lambda name: False
        with self.assertRaisesRegex(RuntimeError, "explicit cloud services"):
            assemble_runtime(character, subsystem, request, {}, unreal, lambda value: None)
        subsystem.request_auto_rigging.assert_not_called()
        subsystem.request_texture_sources.assert_not_called()
        subsystem.build_meta_human.assert_not_called()

    def test_assembly_dispatches_configured_pipeline_and_quality_without_changing_defaults(self):
        for settings, pipeline, quality in (
                ({}, "OPTIMIZED", "MEDIUM"),
                ({"pipeline": "optimized", "quality": "high"}, "OPTIMIZED", "HIGH"),
                ({"pipeline": "cinematic", "quality": "low"}, "CINEMATIC", "LOW")):
            with self.subTest(settings=settings):
                unreal, subsystem, _, character, request = self.fixture()
                character.get_editor_property = lambda name: True
                subsystem.can_build_meta_human.return_value = True
                unreal.EditorAssetLibrary.list_assets.return_value = []
                unreal.MetaHumanCharacterEditorBuildParameters = SimpleNamespace
                unreal.MetaHumanDefaultPipelineType = SimpleNamespace(OPTIMIZED="OPTIMIZED", CINEMATIC="CINEMATIC")
                unreal.MetaHumanQualityLevel = SimpleNamespace(HIGH="HIGH", MEDIUM="MEDIUM", LOW="LOW")
                request["profile"]["assembly"] = settings
                result = {}
                with patch("ailias.metahuman.character_features_worker.inspect_runtime", return_value={}):
                    assemble_runtime(character, subsystem, request, result, unreal, lambda value: None)
                parameters = subsystem.build_meta_human.call_args.args[1]
                self.assertEqual(parameters.pipeline_type, pipeline)
                self.assertEqual(parameters.pipeline_quality, quality)
                self.assertEqual(result["assembly_settings"]["quality_applies"], pipeline == "OPTIMIZED")

    def test_combined_edit_and_build_reuses_registration_and_preserves_edits_before_build(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        request["assemble"] = True

        def build(character, subsystem, request, result, unreal, publish):
            self.assertIs(character, candidate)
            self.assertEqual(character.parameter.value, .95)
            self.assertEqual(character.eyes.eye_left.iris.global_tint.r, .2)
            unreal.EditorAssetLibrary.save_loaded_asset.assert_called_once_with(
                candidate, only_if_is_dirty=False)
            unreal.EditorAssetLibrary.save_loaded_asset(candidate, only_if_is_dirty=False)
            unreal.EditorAssetLibrary.save_loaded_asset.side_effect = RuntimeError(
                "Runtime assembly rebound the ARKit skeleton; do not save the editing Character afterward")
            result["blueprint"] = "/Game/Test/Actor.Actor"

        def inspect(*args, **kwargs):
            self.assertEqual(unreal.EditorAssetLibrary.save_loaded_asset.call_count, 2)
            unreal.EditorAssetLibrary.save_loaded_asset.side_effect = RuntimeError(
                "A runtime actor may rebind the ARKit skeleton; do not save the editing Character afterward")
            return {"blueprint": "/Game/Test/Actor.Actor"}

        with patch("ailias.metahuman.character_features_worker.assemble_runtime",
                   side_effect=build) as assembly, \
                patch("ailias.metahuman.character_features_worker.inspect_runtime",
                      side_effect=inspect) as runtime:
            result = run(request, unreal, lambda value: None)
        self.assertEqual(result["status"], "features_saved")
        assembly.assert_called_once()
        runtime.assert_called_once()
        subsystem.try_add_object_to_edit.assert_called_once_with(candidate)
        self.assertEqual(unreal.EditorAssetLibrary.save_loaded_asset.call_count, 2)

    def test_rigged_character_is_saved_before_runtime_skeletons_are_assembled(self):
        unreal, subsystem, _, character, request = self.fixture()
        character.get_editor_property = lambda name: True
        subsystem.can_build_meta_human.return_value = True
        unreal.EditorAssetLibrary.list_assets.return_value = []
        unreal.MetaHumanCharacterEditorBuildParameters = SimpleNamespace
        unreal.MetaHumanDefaultPipelineType = SimpleNamespace(OPTIMIZED="OPTIMIZED")
        unreal.MetaHumanQualityLevel = SimpleNamespace(MEDIUM="MEDIUM")
        operations = []
        unreal.EditorAssetLibrary.save_loaded_asset.side_effect = (
            lambda *args, **kwargs: operations.append("save_character") or True)
        subsystem.build_meta_human.side_effect = (
            lambda *args: operations.append("build_runtime"))
        assemble_runtime(character, subsystem, request, {}, unreal, lambda value: None)
        self.assertEqual(operations, ["save_character", "build_runtime"])

    def test_invalid_feature_state_stops_before_expensive_assembly(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        request["assemble"] = True
        subsystem.commit_eyes_settings.side_effect = lambda character, settings: setattr(
            character.eyes.eye_left.iris.global_tint, "r", 1)
        with patch("ailias.metahuman.character_features_worker.assemble_runtime") as assembly:
            with self.assertRaisesRegex(RuntimeError, "iris properties did not persist"):
                run(request, unreal, lambda value: None)
        assembly.assert_not_called()
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        subsystem.remove_object_to_edit.assert_called_once_with(candidate)

    def test_failed_assembly_retains_feature_checkpoint_and_stage_timing(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        request["assemble"] = True
        receipts = []

        def fail(*args):
            args[3]["stage"] = "assemble_native_runtime"
            args[5](args[3])
            raise RuntimeError("fixture assembly failure")

        with patch("ailias.metahuman.character_features_worker.assemble_runtime", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "fixture assembly failure"):
                run(request, unreal, lambda value: receipts.append(copy.deepcopy(value)))
        self.assertEqual(receipts[-1]["stage"], "assemble_native_runtime")
        self.assertIn("assemble_native_runtime", receipts[-1]["stage_seconds"])
        self.assertGreaterEqual(
            receipts[-1]["worker_seconds"], sum(receipts[-1]["stage_seconds"].values()))
        unreal.EditorAssetLibrary.save_loaded_asset.assert_called_once_with(
            candidate, only_if_is_dirty=False)
        subsystem.remove_object_to_edit.assert_called_once_with(candidate)

    def test_combined_assembly_cannot_silently_change_the_verified_face(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        request["assemble"] = True

        def change_geometry(*args):
            candidate.points[0].x += .1

        with patch("ailias.metahuman.character_features_worker.assemble_runtime",
                   side_effect=change_geometry):
            with self.assertRaisesRegex(RuntimeError, "sculpt state changed"):
                run(request, unreal, lambda value: None)
        unreal.EditorAssetLibrary.save_loaded_asset.assert_called_once()
        subsystem.remove_object_to_edit.assert_called_once_with(candidate)

    def test_receipt_write_failure_still_releases_owned_character(self):
        unreal, subsystem, _, candidate, request = self.fixture()

        def publish(value):
            if value["stage"] == "awaiting_visual_review":
                raise OSError("fixture disk failure")

        with self.assertRaisesRegex(OSError, "fixture disk failure"):
            run(request, unreal, publish)
        subsystem.remove_object_to_edit.assert_called_once_with(candidate)


class RuntimePhotoSkinTests(unittest.TestCase):
    def fixture(self):
        class Texture:
            def __init__(self, path):
                self.path = path

            def get_path_name(self):
                return self.path

        class Material:
            def __init__(self, path, texture):
                self.path, self.texture = path, texture

            def get_name(self):
                return self.path.rsplit("/", 1)[-1]

            def get_path_name(self):
                return self.path

        class Dynamic(Material):
            def __init__(self, parent):
                self.parent = parent
                super().__init__("/Transient/MID_" + parent.get_name(), None)

            def get_texture_parameter_value(self, name):
                return self.parent.texture

        destination = "/Game/Test/Runtime"
        photo = Texture("/Game/Current/PhotoBaseColor.PhotoBaseColor")
        checkerboard = Texture("/Game/Test/Runtime/Face/Baked/T_Head_BC")
        materials = {destination + "/FeatureDraft/Face/Materials/" + name:
                     Material(destination + "/FeatureDraft/Face/Materials/" + name, checkerboard)
                     for name in ("MI_Face_Skin_Baked_LOD0", "MI_Face_Skin_Baked_LOD2", "MI_Teeth_Baked")}
        skin = [material for material in materials.values() if "Skin" in material.get_name()]
        face_materials = [Dynamic(skin[0]), skin[1]]
        face = SimpleNamespace(get_num_materials=lambda: len(face_materials),
                               get_material=lambda index: face_materials[index])
        library = Mock()

        def set_texture(material, name, texture):
            material.texture = texture
            return True

        library.set_material_instance_texture_parameter_value.side_effect = set_texture
        library.get_material_instance_texture_parameter_value.side_effect = lambda material, name: material.texture
        unreal = SimpleNamespace(
            Texture2D=Texture, MaterialInstanceConstant=Material, MaterialInstanceDynamic=Dynamic,
            MaterialEditingLibrary=library,
            load_asset=lambda path: photo if path == photo.path else materials[path])
        return unreal, face, destination, materials, photo, checkerboard

    def test_gray_bake_fails_then_explicit_photo_binding_reaches_dynamic_and_lower_lod_skin(self):
        unreal, face, destination, materials, photo, checkerboard = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "requested photo"):
            inspect_runtime_skin(face, photo.path, unreal)
        binding = bind_runtime_skin(list(materials), destination, photo.path, unreal)
        self.assertEqual(len(binding["materials"]), 2)
        evidence = inspect_runtime_skin(face, photo.path, unreal)
        self.assertEqual([item["basecolor"] for item in evidence], [photo.path, photo.path])
        teeth = next(material for path, material in materials.items() if "Teeth" in path)
        self.assertIs(teeth.texture, checkerboard)
        face.get_material(1).texture = checkerboard
        with self.assertRaisesRegex(RuntimeError, "requested photo"):
            inspect_runtime_skin(face, photo.path, unreal)

    def test_missing_skin_or_rejected_parameter_blocks_instead_of_accepting_an_untextured_actor(self):
        unreal, face, destination, materials, photo, _ = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "no native skin"):
            bind_runtime_skin([], destination, photo.path, unreal)
        face.get_num_materials = lambda: 0
        with self.assertRaisesRegex(RuntimeError, "no verifiable"):
            inspect_runtime_skin(face, photo.path, unreal)
        unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value.side_effect = None
        unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value.return_value = False
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            bind_runtime_skin(list(materials), destination, photo.path, unreal)

    def test_successful_native_readback_wins_even_when_the_setter_returns_false(self):
        for already_bound in (False, True):
            with self.subTest(already_bound=already_bound):
                unreal, face, destination, materials, photo, _ = self.fixture()
                if already_bound:
                    for material in materials.values():
                        material.texture = photo

                def set_but_return_false(material, name, texture):
                    material.texture = texture
                    return False

                unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value.side_effect = (
                    set_but_return_false)
                binding = bind_runtime_skin(list(materials), destination, photo.path, unreal)
                self.assertEqual(len(binding["materials"]), 2)
                self.assertEqual(len(inspect_runtime_skin(face, photo.path, unreal)), 2)

    def test_native_skin_requires_real_basecolour_bindings_at_every_lod(self):
        unreal, face, _, _, _, native_texture = self.fixture()
        evidence = inspect_runtime_skin(face, None, unreal)
        self.assertEqual([item["basecolor"] for item in evidence], [native_texture.path]*2)
        face.get_material(1).texture = None
        with self.assertRaises(RuntimeError):
            inspect_runtime_skin(face, None, unreal)


class FeatureProfileTests(unittest.TestCase):
    def test_missing_truncated_or_nonfinite_iris_evidence_never_passes_verification(self):
        profile = {"version": 1, "eyes": {"global_tint_linear": [.2, .1, .05]}}
        for eye in ({}, {"global_tint_linear": [.2]}, {"global_tint_linear": [float("nan"), .1, .05]}):
            with self.subTest(eye=eye), self.assertRaisesRegex(RuntimeError, "iris"):
                verify_profile(profile, {"eyes": {"eye_left": eye, "eye_right": eye}})

    def test_invalid_profiles_are_rejected_before_native_work(self):
        valid = {"version": 1, "grooms": {"Beard": {"asset": "/Game/Groom.Groom"}}}
        invalid = [
            {"version": 1}, {"version": True}, {"version": 1, "unexpected": True},
            {**valid, "eyes": {"global_tint_linear": [1, float("nan"), 0]}},
            {**valid, "eyes": {"global_saturation": True}},
            {**valid, "eyes": {"primary_color_u": 1.1}},
            {**valid, "eyes": {"secondary_color_v": float("nan")}},
            {**valid, "skin": {}},
            {**valid, "skin": {"u": True}},
            {**valid, "skin": {"v": -1}},
            {**valid, "skin": {"roughness": 1.2}},
            {**valid, "skin": {"invented_pigment": .5}},
            {**valid, "skin": {"face_texture_index": 1.5}},
            {**valid, "skin": {"face_texture_index": True}},
            {**valid, "skin": {"face_texture_index": 153}},
            {**valid, "freckles": {}},
            {**valid, "freckles": {"mask": []}},
            {**valid, "freckles": {"mask": "unknown"}},
            {**valid, "freckles": {"strength": float("nan")}},
            {**valid, "accents": {}},
            {**valid, "accents": {"unknown_region": {"redness": .4}}},
            {**valid, "accents": {"lips": {}}},
            {**valid, "accents": {"lips": {"redness": 1.1}}},
            {**valid, "accents": {"lips": {"roughness": .5}}},
            {**valid, "eyelashes": {"type": "cosmetic"}},
            {**valid, "eyelashes": {"type": []}},
            {**valid, "eyelashes": {"melanin": -1}},
            {**valid, "eyelashes": {"enable_grooms": 1}},
            {**valid, "eyelashes": {"dye_color_linear": [0, 0]}},
            {**valid, "assembly": {"pipeline": "unknown"}},
            {**valid, "assembly": {"pipeline": []}},
            {**valid, "assembly": {"quality": "ultra"}},
            {**valid, "assembly": {"quality": {}}},
            {**valid, "grooms": {"Unknown": {"asset": "/Game/Groom"}}},
            {**valid, "grooms": {"Beard": {"asset": "/Game/Groom.Other"}}},
            {**valid, "grooms": {"Beard": {"asset": "/Game//Groom"}}},
            {**valid, "grooms": {"Beard": {"asset": "/Engine/Groom"}}},
            {**valid, "sculpt": {"reference_landmarks": [[0, 0, 0]],
                               "max_displacement_cm": .5,
                               "deltas": [{"index": 0, "delta_cm": [.4, .4, 0]}]}},
        ]
        for profile in invalid:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_profile(profile)


class FeatureRevisionTests(unittest.TestCase):
    def apply(self, outcome, *, use_working_baseline=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = Store(root / "data", "subject")
        project = root / "Project" / "Review.uproject"
        project.parent.mkdir()
        project.write_text("{}")
        source = "/Game/Original/Character.Character"
        package = character_package(project, source)
        package.parent.mkdir(parents=True)
        package.write_bytes(b"original Character")
        profile = root / "profile.json"
        profile_data = {"version": 1, "eyes": {"global_saturation": 1.5}}
        if outcome.startswith("assembled-skin"):
            profile_data["face_basecolor"] = "/Game/Current/PhotoBaseColor.PhotoBaseColor"
        elif outcome.startswith("assembled-native"):
            profile_data["face_basecolor"] = None
        profile.write_text(json.dumps(profile_data))
        if outcome == "foreign-baseline":
            baseline = store.resolve("facefitting\\working-baseline.json")
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text('{"status": "unrelated_previous_personal_fit"}')
        probe = {"blockers": [], "editor": str(root / "Editor.exe"), "project": str(project)}

        calls = []

        def launch(name, operation, args, directory, environment):
            request = json.loads(Path(environment["AILIAS_FEATURE_REQUEST"]).read_text())
            calls.append(request["operation"])
            self.assertIn("-d3d12", args)
            self.assertNotIn("-NullRHI", args)
            target = character_package(project, request["character"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"candidate Character")
            (directory / "unused-pixels.bin").write_bytes(b"temporary")
            values = {
                "eyes": {"eye_left": {"global_saturation": 1.5},
                         "eye_right": {"global_saturation": 1.5}},
                "landmarks": [[0, 0, 0]], "selections": [], "groom_parameters": {},
            }
            if outcome.startswith(("assembled-skin", "assembled-native")):
                values["face_basecolor"] = profile_data["face_basecolor"]
                values["skin_preview_enabled"] = True
            if request["operation"] == "verify" and outcome == "lost-sculpt":
                values["landmarks"] = [[1, 0, 0]]
            if outcome == "changed-source":
                package.write_bytes(b"unexpected source edit")
            result = {
                "status": {"apply": "features_saved", "build": "runtime_saved",
                           "verify": "features_reloaded"}[request["operation"]],
                "character": request["character"], "values": values,
            }
            if request["operation"] == "apply" and request.get("assemble"):
                blueprint = request["destination"] + "/Runtime/BP_FeatureDraft.BP_FeatureDraft"
                package_path = character_package(project, blueprint)
                package_path.parent.mkdir(parents=True)
                package_path.write_bytes(b"assembled actor")
                result.update(blueprint=blueprint, runtime_assets=[blueprint],
                              runtime={"blueprint": blueprint})
            if request.get("blueprint"):
                result["runtime"] = {"blueprint": request["blueprint"]}
                if outcome == "assembled-lost-runtime":
                    result.pop("runtime")
            if outcome.startswith("assembled-skin"):
                result["runtime"]["skin"] = [{"basecolor": profile_data["face_basecolor"]}]
                if request["operation"] == "verify" and outcome == "assembled-skin-lost":
                    result["runtime"]["skin"] = [{"basecolor": "/Game/Placeholder/Checkerboard"}]
            elif outcome.startswith("assembled-native"):
                result["runtime"]["skin"] = [{"basecolor": "/Game/Native/Face/Baked/T_Head_BC"}]
                if request["operation"] == "verify" and outcome == "assembled-native-lost":
                    result["runtime"]["skin"] = []
            Path(request["receipt"]).write_text(json.dumps(result))
            return {"seconds": 0}

        with patch("ailias.metahuman.character_features.readiness", return_value=probe):
            result = apply_character_features(store, source, profile, Path(probe["editor"]), project,
                                    SimpleNamespace(paths={}, run=launch),
                                    assemble=outcome.startswith("assembled"),
                                    use_working_baseline=use_working_baseline)
        if outcome.startswith("assembled"):
            self.assertEqual(calls, ["apply", "verify"])
        directory = store.resolve("facefitting\\features\\" + result["revision"])
        self.assertFalse((directory / "unused-pixels.bin").exists())
        self.assertEqual(json.loads((directory / "manifest.json").read_text())["status"], result["status"])
        self.assertTrue(store.verify(result["artifacts"]["character"]))
        return result

    def test_success_requires_a_fresh_reload_and_persists_owned_dependencies(self):
        result = self.apply("saved")
        self.assertEqual(result["status"], "features_reloaded_awaiting_visual_review")
        self.assertTrue(result["fresh_editor_reload_verified"])
        self.assertFalse(result["source_character_modified"])
        self.assertFalse(result["likeness_verified"])
        self.assertEqual(result["cleanup"]["removed_files"], 1)

    def test_scratch_run_can_explicitly_exclude_an_old_working_baseline(self):
        result = self.apply("foreign-baseline", use_working_baseline=False)
        self.assertEqual(result["status"], "features_reloaded_awaiting_visual_review")
        self.assertFalse(any("working-baseline" in item["path"] for item in result["input_dependencies"]))

    def test_default_refinement_still_requires_a_valid_existing_baseline(self):
        with self.assertRaisesRegex(ValueError, "verified working baseline"):
            self.apply("foreign-baseline")

    def test_lost_sculpt_and_changed_source_fail_instead_of_claiming_success(self):
        for outcome in ("lost-sculpt", "changed-source"):
            with self.subTest(outcome=outcome):
                result = self.apply(outcome)
                self.assertEqual(result["status"], "failed")
                if outcome == "changed-source":
                    self.assertTrue(result["source_character_modified"])

    def test_optional_assembly_retains_packages_and_requires_fresh_actor_verification(self):
        result = self.apply("assembled")
        self.assertIn("/Runtime/", result["blueprint"])
        self.assertEqual(result["verify"]["runtime"]["blueprint"], result["blueprint"])
        self.assertIn("runtime_0", result["artifacts"])
        self.assertIn("apply_runtime", result)
        self.assertIn("verify_runtime", result)
        self.assertNotIn("build_runtime", result)

    def test_combined_assembly_still_rejects_missing_independent_actor_verification(self):
        result = self.apply("assembled-lost-runtime")
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("fresh_editor_reload_verified", result)

    def test_saved_character_texture_cannot_hide_a_lost_runtime_skin_on_fresh_reload(self):
        self.assertTrue(self.apply("assembled-skin")["fresh_editor_reload_verified"])
        result = self.apply("assembled-skin-lost")
        self.assertEqual(result["status"], "failed")
        self.assertIn("assembled photo skin", result["error"]["message"])
        self.assertNotIn("fresh_editor_reload_verified", result)

    def test_native_skin_cannot_skip_runtime_binding_evidence_on_fresh_reload(self):
        self.assertTrue(self.apply("assembled-native")["fresh_editor_reload_verified"])
        result = self.apply("assembled-native-lost")
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("fresh_editor_reload_verified", result)


if __name__ == "__main__":
    unittest.main()
