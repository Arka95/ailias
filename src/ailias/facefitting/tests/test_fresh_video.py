"""Fresh-native boundaries without loading models, hardware, real media or UE."""

from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ailias.facefitting.fresh_video_worker import run
from ailias.facefitting.pipeline import export_obj
from ailias.facefitting.unreal_conform_worker import geometry_delta


class NativeObjExportTests(unittest.TestCase):
    def test_native_obj_contains_real_per_corner_uv_indices_including_seams(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]])
        triangles = np.array([[0, 1, 2], [2, 1, 0]])
        uvs = np.array([[[0., 0], [1., 0], [0., 1]], [[.2, 1.], [1., .2], [.1, .1]]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "head.obj"
            export_obj(path, vertices, triangles, triangle_uvs=uvs)
            lines = path.read_text().splitlines()
        self.assertEqual(len([s for s in lines if s.startswith("vt ")]), 6)
        self.assertEqual([s for s in lines if s.startswith("f ")],
                         ["f 1/1 2/2 3/3", "f 3/4 2/5 1/6"])
        actual = np.array([[float(v) for v in s.split()[1:]] for s in lines if s.startswith("vt ")])
        np.testing.assert_allclose(actual, uvs.reshape(-1, 2))

    def test_invalid_uv_topology_is_rejected_before_writing_a_native_obj(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "head.obj"
            with self.assertRaisesRegex(ValueError, "per-corner topology"):
                export_obj(path, np.zeros((3, 3)), np.array([[0, 1, 2]]),
                           triangle_uvs=np.zeros((1, 2, 2)))
            self.assertFalse(path.exists())


class FreshVideoWorkerTests(unittest.TestCase):
    def fixture(self):
        class Character:
            pass

        character = Character()
        subsystem, library = Mock(), Mock()
        subsystem.try_add_object_to_edit.return_value = True
        subsystem.get_face_landmarks.return_value = [SimpleNamespace(x=1, y=2, z=3)]
        library.list_assets.return_value = ["/Game/Occupied"]
        unreal = SimpleNamespace(
            load_asset=lambda name: character, MetaHumanCharacter=Character,
            MetaHumanCharacterEditorSubsystem=object,
            get_editor_subsystem=lambda cls: subsystem,
            EditorAssetLibrary=library, AssetToolsHelpers=Mock())
        return unreal, subsystem

    def test_reload_checks_the_requested_new_character_without_creating_or_saving(self):
        unreal, subsystem = self.fixture()
        result = run({"operation": "verify", "character": "/Game/Fresh.Head",
                      "expected": [[1., 2., 3.]]}, unreal)
        self.assertEqual(result["status"], "fresh_character_reloaded")
        self.assertFalse(result["likeness_verified"])
        subsystem.remove_object_to_edit.assert_called_once()
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        unreal.AssetToolsHelpers.get_asset_tools.assert_not_called()

    def test_lost_geometry_on_reload_is_a_failure_and_releases_editing(self):
        unreal, subsystem = self.fixture()
        receipts = []
        with self.assertRaisesRegex(RuntimeError, "differs from"):
            run({"operation": "verify", "character": "/Game/Fresh.Head",
                 "expected": [[9., 2., 3.]]}, unreal,
                lambda value: receipts.append(dict(value)))
        subsystem.remove_object_to_edit.assert_called_once()
        self.assertEqual(receipts[-1]["maximum_distance_cm"], 8)
        self.assertEqual(receipts[-1]["landmarks"], [[1., 2., 3.]])

    def test_nonfinite_landmarks_never_pass_reload_verification(self):
        unreal, subsystem = self.fixture()
        subsystem.get_face_landmarks.return_value = [SimpleNamespace(x=float("nan"), y=2, z=3)]
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            run({"operation": "verify", "character": "/Game/Fresh.Head",
                 "expected": [[1., 2., 3.]]}, unreal)
        subsystem.remove_object_to_edit.assert_called_once()

    def surface_verification(self, changed_surface):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        reference = root / "native-geometry-before.json"
        expected = {"vertices": [[0., 0, 0], [1., 0, 0], [0., 1, 0]], "indices": [0, 1, 2]}
        reference.write_text(json.dumps(expected))

        def export(character, subsystem, unreal, destination, output):
            output.write_text(json.dumps(changed_surface or expected))
            return {"path": str(output), "vertex_count": 3, "mesh": "/Game/Current/TransientHead"}

        module = SimpleNamespace(export_saved_geometry=export, geometry_delta=geometry_delta)
        unreal, subsystem = self.fixture()
        request = {
            "operation": "verify", "character": "/Game/Fresh.Head", "expected": [[9., 2., 3.]],
            "expected_surface": str(reference), "surface_destination": "/Game/Fresh/Surface",
            "conform_worker": "worker.py",
        }
        return unreal, subsystem, module, request

    def test_surface_persistence_is_not_rejected_solely_for_recomputed_sculpt_controls(self):
        unreal, subsystem, module, request = self.surface_verification(None)
        with patch("ailias.facefitting.fresh_video_worker.load_conform_worker", return_value=module):
            result = run(request, unreal)
        self.assertEqual(result["status"], "fresh_character_reloaded")
        self.assertEqual(result["surface_delta"]["maximum_distance_cm"], 0)
        self.assertEqual(result["maximum_distance_cm"], 8)
        subsystem.remove_object_to_edit.assert_called_once()

    def test_a_real_surface_change_still_fails_with_independently_exported_geometry(self):
        unreal, subsystem, module, request = self.surface_verification({
            "vertices": [[0., 0, 0], [1.01, 0, 0], [0., 1, 0]], "indices": [0, 1, 2]})
        with patch("ailias.facefitting.fresh_video_worker.load_conform_worker", return_value=module):
            with self.assertRaisesRegex(RuntimeError, "surface changed"):
                run(request, unreal)
        subsystem.remove_object_to_edit.assert_called_once()

    def test_native_uvs_are_exported_before_the_verifying_editor_releases_its_unsaved_mesh(self):
        unreal, subsystem, module, request = self.surface_verification(None)
        uv = Mock()
        uv.export_surface.return_value = {"status": "native_skin_uvs_exported"}
        request.update(surface_worker="uv-worker.py", native_uv_output="current-uvs.json",
                       canonical_to_target={"rotation": [], "translation_cm": []})
        with patch("ailias.facefitting.fresh_video_worker.load_conform_worker", side_effect=[module, uv]):
            result = run(request, unreal)
        self.assertEqual(result["native_skin_uvs"]["status"], "native_skin_uvs_exported")
        arguments, = uv.export_surface.call_args.args[:1]
        self.assertEqual(arguments["mesh"], "/Game/Current/TransientHead")
        self.assertEqual(arguments["canonical_to_target"], request["canonical_to_target"])
        subsystem.remove_object_to_edit.assert_called_once()

    def test_failed_uv_export_does_not_promote_geometry_only_to_complete_native_verification(self):
        unreal, subsystem, module, request = self.surface_verification(None)
        uv = Mock()
        uv.export_surface.side_effect = RuntimeError("native UVs missing")
        request.update(surface_worker="uv-worker.py", native_uv_output="current-uvs.json",
                       canonical_to_target={})
        with patch("ailias.facefitting.fresh_video_worker.load_conform_worker", side_effect=[module, uv]):
            with self.assertRaisesRegex(RuntimeError, "native UVs missing"):
                run(request, unreal)
        subsystem.remove_object_to_edit.assert_called_once()

    def test_occupied_namespace_never_reuses_existing_assets(self):
        unreal, subsystem = self.fixture()
        with self.assertRaisesRegex(ValueError, "refusing to reuse"):
            run({"operation": "create", "destination": "/Game/Occupied"}, unreal)
        unreal.AssetToolsHelpers.get_asset_tools.assert_not_called()
        subsystem.try_add_object_to_edit.assert_not_called()
