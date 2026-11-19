"""Fresh-native boundaries without loading models, hardware, real media or UE."""

from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ailias.metahuman.character_creation_worker import execute_character_request
from ailias.metahuman.head_conforming_worker import geometry_delta


class CharacterCreationTests(unittest.TestCase):
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
        result = execute_character_request({"operation": "verify", "character": "/Game/Fresh.Head",
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
            execute_character_request({"operation": "verify", "character": "/Game/Fresh.Head",
                 "expected": [[9., 2., 3.]]}, unreal,
                lambda value: receipts.append(dict(value)))
        subsystem.remove_object_to_edit.assert_called_once()
        self.assertEqual(receipts[-1]["maximum_distance_cm"], 8)
        self.assertEqual(receipts[-1]["landmarks"], [[1., 2., 3.]])

    def test_nonfinite_landmarks_never_pass_reload_verification(self):
        unreal, subsystem = self.fixture()
        subsystem.get_face_landmarks.return_value = [SimpleNamespace(x=float("nan"), y=2, z=3)]
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            execute_character_request({"operation": "verify", "character": "/Game/Fresh.Head",
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
        with patch("ailias.metahuman.character_creation_worker.load_worker_module", return_value=module):
            result = execute_character_request(request, unreal)
        self.assertEqual(result["status"], "fresh_character_reloaded")
        self.assertEqual(result["surface_delta"]["maximum_distance_cm"], 0)
        self.assertEqual(result["maximum_distance_cm"], 8)
        subsystem.remove_object_to_edit.assert_called_once()

    def test_a_real_surface_change_still_fails_with_independently_exported_geometry(self):
        unreal, subsystem, module, request = self.surface_verification({
            "vertices": [[0., 0, 0], [1.01, 0, 0], [0., 1, 0]], "indices": [0, 1, 2]})
        with patch("ailias.metahuman.character_creation_worker.load_worker_module", return_value=module):
            with self.assertRaisesRegex(RuntimeError, "surface changed"):
                execute_character_request(request, unreal)
        subsystem.remove_object_to_edit.assert_called_once()

    def test_native_uvs_are_exported_before_the_verifying_editor_releases_its_unsaved_mesh(self):
        unreal, subsystem, module, request = self.surface_verification(None)
        uv = Mock()
        uv.export_skin_uv_mesh.return_value = {"status": "native_skin_uvs_exported"}
        request.update(surface_worker="uv-worker.py", native_uv_output="current-uvs.json",
                       canonical_to_target={"rotation": [], "translation_cm": []})
        with patch("ailias.metahuman.character_creation_worker.load_worker_module", side_effect=[module, uv]):
            result = execute_character_request(request, unreal)
        self.assertEqual(result["native_skin_uvs"]["status"], "native_skin_uvs_exported")
        arguments, = uv.export_skin_uv_mesh.call_args.args[:1]
        self.assertEqual(arguments["mesh"], "/Game/Current/TransientHead")
        self.assertEqual(arguments["canonical_to_target"], request["canonical_to_target"])
        subsystem.remove_object_to_edit.assert_called_once()

    def test_failed_uv_export_does_not_promote_geometry_only_to_complete_native_verification(self):
        unreal, subsystem, module, request = self.surface_verification(None)
        uv = Mock()
        uv.export_skin_uv_mesh.side_effect = RuntimeError("native UVs missing")
        request.update(surface_worker="uv-worker.py", native_uv_output="current-uvs.json",
                       canonical_to_target={})
        with patch("ailias.metahuman.character_creation_worker.load_worker_module", side_effect=[module, uv]):
            with self.assertRaisesRegex(RuntimeError, "native UVs missing"):
                execute_character_request(request, unreal)
        subsystem.remove_object_to_edit.assert_called_once()

    def test_occupied_namespace_never_reuses_existing_assets(self):
        unreal, subsystem = self.fixture()
        with self.assertRaisesRegex(ValueError, "refusing to reuse"):
            execute_character_request({"operation": "create", "destination": "/Game/Occupied"}, unreal)
        unreal.AssetToolsHelpers.get_asset_tools.assert_not_called()
        subsystem.try_add_object_to_edit.assert_not_called()
