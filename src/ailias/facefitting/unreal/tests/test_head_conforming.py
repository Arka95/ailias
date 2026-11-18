"""Test local Character solve/commit contracts with a fake editor and no native/model execution."""

import json
import itertools
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from ailias.facefitting.unreal.head_conforming import conform_head
from ailias.facefitting.unreal.head_conforming_worker import run
from ailias.facefitting.blender.tracking_view_worker import TRACKING_FOV_DEGREES, frontal_frame
from ailias.storage import Store, atomic_json


class CharacterConformTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        camera = root / "camera.json"
        camera.write_text(json.dumps({
            "projection": "perspective", "image_size": [1, 1],
            "position_gnm": [0, .25, .82], "forward_gnm": [0, 0, -1],
            "up_gnm": [0, 1, 0], "horizontal_fov_degrees": 30,
        }))
        rgba = root / "rgba.bin"
        rgba.write_bytes(bytes((100, 80, 60, 255)))

        class Mesh:
            pass

        class Character:
            rigged = True

            def get_path_name(self):
                return "/Game/Test/HeadOnlyDraft.HeadOnlyDraft"

            def get_editor_property(self, name):
                if name == "has_face_dna_blendshapes":
                    return self.rigged
                raise AttributeError(name)

        character = Character()
        point = SimpleNamespace(x=1, y=2, z=3)
        subsystem = Mock()
        subsystem.get_mesh_data_for_conforming.return_value = ([point]*3, [0, 1, 2])
        subsystem.track_face_landmarks_from_image.return_value = {
            "nose": SimpleNamespace(tracking_points=[SimpleNamespace(x=.5, y=.5)])}
        subsystem.try_add_object_to_edit.return_value = True
        subsystem.conform_to_target_meshes.return_value = True
        changed = [SimpleNamespace(x=2, y=3, z=4)]
        subsystem.get_face_landmarks.side_effect = [[point], changed, changed, changed, changed, changed]
        library, tools = Mock(), Mock()
        library.list_assets.return_value = []
        library.save_loaded_asset.return_value = True
        tools.create_asset.return_value = character
        unreal = SimpleNamespace(
            SystemLibrary=SimpleNamespace(get_engine_version=lambda: "5.8.4-fixture"),
            EditorAssetLibrary=library, StaticMesh=Mesh, MetaHumanCharacter=Character,
            load_asset=lambda name: Mesh(), MetaHumanCharacterEditorSubsystem=object,
            get_editor_subsystem=lambda cls: subsystem, Color=lambda *args: args,
            MetaHumanCharacterFactoryNew=SimpleNamespace,
            AssetToolsHelpers=SimpleNamespace(get_asset_tools=lambda: tools),
            ConformTargetMesh=SimpleNamespace, TargetPartsType=SimpleNamespace(HEAD_ONLY="head"),
            ConformTargetParams=lambda: SimpleNamespace(body_conform_solve_settings=SimpleNamespace()),
            IntPoint=lambda *args: args, MinimalViewInfo=SimpleNamespace,
            Vector=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
            Vector3f=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
            MathLibrary=SimpleNamespace(make_rot_from_xz=Mock(return_value="front_rotation")),
            CameraProjectionMode=SimpleNamespace(PERSPECTIVE="perspective"),
            MetaHumanCharacterTargetMeshKey=SimpleNamespace,
        )
        return unreal, subsystem, tools, {
            "mesh": "/Game/Head", "destination": "/Game/Test",
            "camera": str(camera), "rgba": str(rgba),
        }

    def test_native_success_commits_both_states_saves_and_releases_without_cloud(self):
        unreal, subsystem, _, request = self.fixture()
        result = run(request, unreal, lambda value: None)
        self.assertEqual(result["status"], "head_conformed_awaiting_review")
        self.assertFalse(result["assembled"])
        self.assertFalse(result["appearance_transferred"])
        parameters = subsystem.conform_to_target_meshes.call_args.args[2]
        self.assertEqual(parameters.conform_target_mesh.target_parts_type, "head")
        self.assertEqual(parameters.camera_view_info.location.y, 82)
        self.assertEqual(parameters.camera_view_info.location.z, 25)
        target_vertex = parameters.conform_target_mesh.head_vertices[0]
        self.assertEqual((target_vertex.x, target_vertex.y, target_vertex.z), (-2, 1, 3))
        subsystem.commit_face_state.assert_called_once()
        subsystem.commit_body_state.assert_called_once()
        subsystem.commit_posed_state_as_a_pose.assert_called_once()
        subsystem.remove_object_to_edit.assert_called_once()
        subsystem.request_auto_rigging.assert_not_called()
        subsystem.request_texture_sources.assert_not_called()
        self.assertEqual(result["face_landmarks_saved"], [[2., 3., 4.]])
        self.assertEqual(result["save_geometry"]["maximum_distance_cm"], 0)

    def test_body_commit_cannot_silently_replace_the_conformed_shape(self):
        unreal, subsystem, _, request = self.fixture()
        original = [SimpleNamespace(x=1, y=2, z=3)]
        changed = [SimpleNamespace(x=2, y=3, z=4)]
        subsystem.get_face_landmarks.side_effect = [original, changed, changed, changed, original]
        receipts = []
        with self.assertRaisesRegex(RuntimeError, "commit changed"):
            run(request, unreal, lambda value: receipts.append(dict(value)))
        self.assertGreater(receipts[-1]["commit_geometry"]["body"]["maximum_distance_cm"], 1)
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        subsystem.remove_object_to_edit.assert_called_once()

    def test_save_failure_releases_character_and_never_claims_saved_geometry(self):
        unreal, subsystem, _, request = self.fixture()
        unreal.EditorAssetLibrary.save_loaded_asset.side_effect = OSError("disk unavailable")
        receipts = []
        with self.assertRaisesRegex(OSError, "disk unavailable"):
            run(request, unreal, lambda value: receipts.append(dict(value)))
        self.assertNotIn("face_landmarks_saved", receipts[-1])
        subsystem.remove_object_to_edit.assert_called_once()

    def test_cleanup_failure_does_not_replace_a_native_solve_failure(self):
        unreal, subsystem, _, request = self.fixture()
        subsystem.conform_to_target_meshes.return_value = False
        subsystem.remove_object_to_edit.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "solver returned false") as caught:
            run(request, unreal, lambda value: None)
        self.assertIn("cleanup also failed", caught.exception.__notes__[0])

    def test_native_failure_never_commits_or_saves_and_releases_editor_state(self):
        unreal, subsystem, _, request = self.fixture()
        subsystem.conform_to_target_meshes.return_value = False
        with self.assertRaisesRegex(RuntimeError, "returned false"):
            run(request, unreal, lambda value: None)
        subsystem.commit_face_state.assert_not_called()
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        subsystem.remove_object_to_edit.assert_called_once()

    def test_existing_character_is_duplicated_before_refinement_not_edited_in_place(self):
        unreal, subsystem, tools, request = self.fixture()
        character = unreal.MetaHumanCharacter()
        candidate = unreal.MetaHumanCharacter()
        request["source_character"] = "/Game/Existing.Head"
        unreal.load_asset = lambda name: character if name == request["source_character"] else unreal.StaticMesh()
        unreal.EditorAssetLibrary.duplicate_asset.return_value = candidate
        subsystem.is_object_added_for_editing.return_value = False
        subsystem.remove_face_rig.side_effect = lambda value: setattr(value, "rigged", False)
        result = run(request, unreal, lambda value: None)
        self.assertEqual(result["source_character"], request["source_character"])
        unreal.EditorAssetLibrary.duplicate_asset.assert_called_once_with(
            request["source_character"], "/Game/Test/HeadOnlyDraft")
        tools.create_asset.assert_not_called()
        subsystem.try_add_object_to_edit.assert_called_once_with(candidate)
        subsystem.remove_face_rig.assert_called_once_with(candidate)
        self.assertTrue(character.rigged)
        self.assertFalse(candidate.rigged)
        self.assertTrue(result["inherited_face_rig_removed"])

    def test_failed_tracking_does_not_create_or_register_a_character(self):
        unreal, subsystem, tools, request = self.fixture()
        subsystem.track_face_landmarks_from_image.return_value = None
        with self.assertRaisesRegex(RuntimeError, "tracking failed"):
            run(request, unreal, lambda value: None)
        tools.create_asset.assert_not_called()
        subsystem.try_add_object_to_edit.assert_not_called()

    def test_inherited_rig_that_cannot_be_removed_blocks_geometry_editing(self):
        unreal, subsystem, _, request = self.fixture()
        original, candidate = unreal.MetaHumanCharacter(), unreal.MetaHumanCharacter()
        request["source_character"] = "/Game/Existing.Head"
        unreal.load_asset = lambda name: original if name == request["source_character"] else unreal.StaticMesh()
        unreal.EditorAssetLibrary.duplicate_asset.return_value = candidate
        subsystem.is_object_added_for_editing.return_value = False
        with self.assertRaisesRegex(RuntimeError, "must be removed"):
            run(request, unreal, lambda value: None)
        subsystem.conform_to_target_meshes.assert_not_called()
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        subsystem.remove_object_to_edit.assert_called_once_with(candidate)


class ConformPersistenceTests(unittest.TestCase):
    def conform(self, outcome, baseline_mode=None, photo=False, existing=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = Store(root / "data", "subject")
        imported_path = store.resolve("facefitting\\unreal\\imported\\manifest.json")
        imported_path.parent.mkdir(parents=True)
        source = imported_path.parent / "head.obj"
        source.write_text("synthetic head")
        scene = imported_path.parent / "scene.blend"
        scene.write_bytes(b"synthetic scene")
        selected_scene = imported_path.parent / "working.blend"
        if baseline_mode:
            selected_scene.write_bytes(b"synthetic selected baseline")
            baseline_path = store.resolve("facefitting\\working-baseline.json")
            atomic_json(baseline_path, {
                "status": "owner_selected_working_baseline", "revision": "selected",
                "appearance_revision": "another" if baseline_mode == "wrong-appearance" else "appearance",
                "artifacts": {"scene": store.reference_artifact(selected_scene),
                              "source_scene": store.reference_artifact(scene)}})
            if baseline_mode == "tampered":
                selected_scene.write_bytes(b"changed selected baseline")
        project = root / "Project" / "Review.uproject"
        project.parent.mkdir()
        project.write_text("{}")
        if existing:
            original = project.parent / "Content" / "Existing" / "Head.uasset"
            original.parent.mkdir(parents=True)
            original.write_bytes(b"unchanged original native Character")
        imported = {"status": "static_head_imported_for_review", "revision": "appearance",
                    "source": store.reference_artifact(source), "artifacts": {},
                    "readiness": {"project": str(project)},
                    "imported": {"mesh_geometry": [{"asset": "/Game/Head.Head"}]}}
        atomic_json(imported_path, imported)
        probe = {"blockers": [], "editor": str(root / "Editor.exe"), "project": str(project)}

        def launch(name, operation, args, directory, environment=None):
            if name == "blender":
                self.assertFalse(photo, "Photo contour mode must not launch Blender")
                expected = selected_scene if baseline_mode else scene
                self.assertIn(expected.resolve(), args)
                tracking_request = json.loads(Path(args[-1]).read_text())
                self.assertEqual(tracking_request["camera_mode"], "frontal" if baseline_mode else "retained")
                Image.new("RGBA", (1, 1)).save(directory / "tracking-view.png")
                atomic_json(directory / "tracking-camera.json", {})
            else:
                request = json.loads(Path(environment["AILIAS_CONFORM_REQUEST"]).read_text())
                if existing:
                    self.assertIn("-d3d12", args)
                    self.assertNotIn("-NullRHI", args)
                    self.assertEqual(request["source_character"], "/Game/Existing/Head.Head")
                self.assertEqual(Path(request["rgba"]).stat().st_size, 4)
                if outcome == "malformed":
                    Path(request["receipt"]).write_text("{")
                    return {"seconds": 0}
                package = request["destination"] + "/HeadOnlyDraft.HeadOnlyDraft"
                if outcome == "saved":
                    target = project.parent / "Content" / Path(*request["destination"].split("/")[2:])
                    target.mkdir(parents=True)
                    (target / "HeadOnlyDraft.uasset").write_bytes(b"synthetic package")
                atomic_json(Path(request["receipt"]), {
                    "status": "head_conformed_awaiting_review", "assets": [package],
                    "character": package, "face_landmarks_before": [[0, 0, 0]],
                    "face_landmarks_after": [[1, 0, 0]]})
            return {"seconds": 0}

        def prepare_photo(selected_store, fit_job, face_id, directory):
            self.assertEqual((fit_job, face_id), ("fit", "photo"))
            Image.new("RGBA", (1, 1)).save(directory / "tracking-view.png")
            atomic_json(directory / "tracking-camera.json", {})
            return {"input_dependencies": [store.reference_artifact(source)], "geometry_refitted": False}

        appearance = {"integrity_problems": [], "fit_job": "fit",
                      "artifacts": {"scene": store.reference_artifact(scene)}}
        with patch("ailias.facefitting.unreal.head_conforming.inspect_head_appearance", return_value=appearance), \
                patch("ailias.facefitting.unreal.head_conforming.readiness", return_value=probe), \
                patch("ailias.facefitting.geometry.photo_camera.prepare_photo_tracking", side_effect=prepare_photo):
            result = conform_head(store, "imported", Path(probe["editor"]), project,
                                  SimpleNamespace(blender_device="CUDA"),
                                  SimpleNamespace(paths={}, run=launch),
                                  reference_face_id="photo" if photo else None,
                                  source_character="/Game/Existing/Head.Head" if existing else None)
        job = store.resolve("facefitting\\conforms\\" + result["revision"])
        self.assertFalse((job / "tracking-rgba8.bin").exists())
        self.assertEqual(json.loads((job / "manifest.json").read_text())["status"], result["status"])
        return store, result

    def test_success_requires_a_saved_package_and_preserves_auditable_camera_evidence(self):
        store, result = self.conform("saved")
        self.assertEqual(result["status"], "head_conformed_awaiting_review")
        self.assertTrue(store.verify(result["artifacts"]["character_0"]))
        self.assertTrue(store.verify(result["artifacts"]["tracking-camera.json"]))
        self.assertFalse(result["likeness_verified"])

    def test_selected_baseline_scene_drives_a_new_frontal_tracking_view(self):
        store, result = self.conform("saved", "valid")
        self.assertEqual(result["status"], "head_conformed_awaiting_review")
        self.assertEqual(result["working_baseline"]["revision"], "selected")
        self.assertFalse(result["working_baseline"]["baseline_scene_modified"])
        self.assertTrue(all(store.verify(reference) for reference in result["input_dependencies"]))

    def test_existing_rigged_source_uses_real_rhi_and_is_pinned_without_overwrite(self):
        store, result = self.conform("saved", "valid", photo=True, existing=True)
        self.assertEqual(result["status"], "head_conformed_awaiting_review")
        self.assertEqual(result["source_character"], "/Game/Existing/Head.Head")
        self.assertTrue(all(store.verify(reference) for reference in result["input_dependencies"]))

    def test_original_photo_mode_bypasses_blender_and_keeps_source_dependencies(self):
        store, result = self.conform("saved", "valid", photo=True)
        self.assertEqual(result["status"], "head_conformed_awaiting_review")
        self.assertEqual(result["tracking_source"], "original_photo")
        self.assertEqual(result["working_baseline"]["tracking_camera_mode"], "fitted_photo")
        self.assertFalse(result["photo_tracking"]["geometry_refitted"])
        self.assertTrue(all(store.verify(item) for item in result["input_dependencies"]))

    def test_changed_or_unrelated_baseline_is_rejected_before_native_work(self):
        for mode in ("tampered", "wrong-appearance"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "baseline"):
                self.conform("saved", mode)

    def test_native_success_without_package_is_failure_and_cleans_temporary_pixels(self):
        _, result = self.conform("missing_package")
        self.assertEqual(result["status"], "failed")

    def test_malformed_native_receipt_retains_failure_and_does_not_mask_cleanup(self):
        _, result = self.conform("malformed")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["native_receipt_unreadable"])


class FrontalTrackingTests(unittest.TestCase):
    def test_frontal_camera_contains_the_entire_head_with_margin_and_tracks_translation(self):
        for translation in ((0, 0, 0), (1, -2, 3)):
            bounds = [[value+shift for value, shift in zip(point, translation)]
                      for point in itertools.product((-1, 1), (-2, 2), (-.5, .5))]
            target, position = frontal_frame(bounds)
            self.assertEqual(target, list(translation))
            tangent = math.tan(math.radians(TRACKING_FOV_DEGREES)/2)
            for x, y, z in bounds:
                depth = position[2]-z
                self.assertGreater(depth, 0)
                self.assertLessEqual(abs((x-position[0])/depth/tangent), 1/1.2 + 1e-10)
                self.assertLessEqual(abs((y-position[1])/depth/tangent), 1/1.2 + 1e-10)

    def test_invalid_head_bounds_cannot_produce_a_tracking_camera(self):
        for bounds in ([(0, 0, 0)]*8, [(0, 0, 0), (float("nan"), 1, 1)]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                frontal_frame(bounds)


if __name__ == "__main__":
    unittest.main()
