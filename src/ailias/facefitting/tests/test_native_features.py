"""Verify native edits and immutable revision handling without loading Unreal or models."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ailias.facefitting.native_features import apply_features, character_package
from ailias.facefitting.native_features_worker import assemble_runtime, run, validate_profile
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
                    global_tint=SimpleNamespace(r=1., g=1., b=1.), global_saturation=1.7)
                self.eyes = SimpleNamespace(
                    eye_left=SimpleNamespace(iris=copy.deepcopy(iris)),
                    eye_right=SimpleNamespace(iris=copy.deepcopy(iris)))
                self.skin = SimpleNamespace(texture_material_overrides=SimpleNamespace(
                    enable_texture_overrides=False, texture_overrides=SimpleNamespace(face={})))

            def get_path_name(self):
                return self.path

            def get_editor_property(self, name):
                return self.eyes if name == "eyes_settings" else self.skin

        source, candidate = Character(source_path), Character(candidate_path)
        wardrobe = Wardrobe()
        subsystem, library = Mock(), Mock()
        subsystem.is_object_added_for_editing.return_value = False
        subsystem.try_add_object_to_edit.return_value = True
        subsystem.get_face_landmarks.side_effect = lambda character: character.points
        subsystem.get_preview_collection.side_effect = lambda character: character.internal_collection
        subsystem.commit_eyes_settings.side_effect = lambda character, settings: setattr(
            character, "eyes", settings)
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

    def test_combined_edit_and_build_reuses_registration_and_preserves_edits_before_build(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        request["assemble"] = True

        def build(character, subsystem, request, result, unreal, publish):
            self.assertIs(character, candidate)
            self.assertEqual(character.parameter.value, .95)
            self.assertEqual(character.eyes.eye_left.iris.global_tint.r, .2)
            unreal.EditorAssetLibrary.save_loaded_asset.assert_called_once_with(
                candidate, only_if_is_dirty=False)

        with patch("ailias.facefitting.native_features_worker.assemble_runtime",
                   side_effect=build) as assembly:
            result = run(request, unreal, lambda value: None)
        self.assertEqual(result["status"], "features_saved")
        assembly.assert_called_once()
        subsystem.try_add_object_to_edit.assert_called_once_with(candidate)
        self.assertEqual(unreal.EditorAssetLibrary.save_loaded_asset.call_count, 2)

    def test_invalid_feature_state_stops_before_expensive_assembly(self):
        unreal, subsystem, _, candidate, request = self.fixture()
        request["assemble"] = True
        subsystem.commit_eyes_settings.side_effect = lambda character, settings: setattr(
            character.eyes.eye_left.iris.global_tint, "r", 1)
        with patch("ailias.facefitting.native_features_worker.assemble_runtime") as assembly:
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

        with patch("ailias.facefitting.native_features_worker.assemble_runtime", side_effect=fail):
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

        with patch("ailias.facefitting.native_features_worker.assemble_runtime",
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


class FeatureProfileTests(unittest.TestCase):
    def test_invalid_profiles_are_rejected_before_native_work(self):
        valid = {"version": 1, "grooms": {"Beard": {"asset": "/Game/Groom.Groom"}}}
        invalid = [
            {"version": 1}, {"version": True}, {"version": 1, "unexpected": True},
            {**valid, "eyes": {"global_tint_linear": [1, float("nan"), 0]}},
            {**valid, "eyes": {"global_saturation": True}},
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
    def apply(self, outcome):
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
        profile.write_text(json.dumps({"version": 1, "eyes": {"global_saturation": 1.5}}))
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
            Path(request["receipt"]).write_text(json.dumps(result))
            return {"seconds": 0}

        with patch("ailias.facefitting.native_features.readiness", return_value=probe):
            result = apply_features(store, source, profile, Path(probe["editor"]), project,
                                    SimpleNamespace(paths={}, run=launch),
                                    assemble=outcome.startswith("assembled"))
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


if __name__ == "__main__":
    unittest.main()
