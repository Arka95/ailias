import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from ailias.metahuman.sculpt_response_worker import run
from ailias.metahuman.sculpt_response import measure_response
from ailias.storage import Store


class NativeResponseTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        class Character:
            def get_editor_property(self, name):
                return False

        class Mesh:
            pass

        original = [[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]]
        state = {"live": copy.deepcopy(original), "committed": copy.deepcopy(original)}
        character, mesh = Character(), Mesh()
        subsystem = Mock()
        subsystem.try_add_object_to_edit.return_value = True
        vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        subsystem.get_face_landmarks.side_effect = lambda value: [vector(*p) for p in state["live"]]
        subsystem.get_face_model_coefficients.side_effect = lambda value: [
            v for point in state["live"] for v in point]

        def restore(value, coefficients):
            state["live"] = [coefficients[i:i+3] for i in range(0, len(coefficients), 3)]

        def translate(value, indices, deltas):
            for index, delta in zip(indices, deltas):
                state["live"][index] = [
                    a+.5*b for a, b in zip(state["live"][index], (delta.x, delta.y, delta.z))]

        subsystem.set_face_model_coefficients.side_effect = restore
        subsystem.translate_face_landmarks.side_effect = translate
        subsystem.commit_face_state.side_effect = lambda value: state.update(
            committed=copy.deepcopy(state["live"]))
        subsystem.get_mesh_data_for_conforming.side_effect = lambda value: (
            [vector(*p) for p in state["committed"]], [0, 1, 2])
        library = Mock()
        library.list_assets.side_effect = lambda path, **kwargs: (
            ["/Game/Response/Mesh/Head"] if path.endswith("/Mesh") else [])
        library.duplicate_asset.return_value = character
        unreal = SimpleNamespace(
            MetaHumanCharacterEditorSubsystem=object,
            get_editor_subsystem=lambda cls: subsystem,
            EditorAssetLibrary=library, MetaHumanCharacter=Character,
            MetaHumanGeometryExportParams=SimpleNamespace,
            MetaHumanCharacterExportBlueprintLibrary=SimpleNamespace(export_geometry=Mock()),
            SkeletalMesh=Mesh, load_asset=lambda path: mesh, Vector=vector,
        )
        request = {"step_cm": .05, "output": str(root), "destination": "/Game/Response",
                   "character": "/Game/Original.Head", "indices": [1]}
        return unreal, subsystem, state, original, request

    def test_measures_real_committed_mesh_and_restores_opaque_state(self):
        unreal, subsystem, state, original, request = self.fixture()
        result = run(request, unreal, lambda value: None)
        self.assertEqual(result["status"], "native_response_measured")
        self.assertFalse(result["likeness_verified"])
        self.assertEqual(len(result["responses"]), 3)
        values = np.fromfile(Path(request["output"]) / "responses.f32", dtype="<f4").reshape(3, 3, 3)
        expected = np.zeros((3, 3, 3))
        expected[:, 1, :] = np.eye(3)*.5
        np.testing.assert_allclose(values, expected, atol=1e-6)
        self.assertEqual(state["live"], original)
        self.assertEqual(state["committed"], original)
        subsystem.remove_object_to_edit.assert_called_once()
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        for call in unreal.MetaHumanCharacterExportBlueprintLibrary.export_geometry.call_args_list:
            self.assertEqual(call.args[1].project_path, "/Game/Response/Mesh")
            self.assertTrue(call.args[1].overwrite_existing_assets)

    def test_topology_change_is_rejected_and_reference_is_restored(self):
        unreal, subsystem, state, original, request = self.fixture()
        get_mesh = subsystem.get_mesh_data_for_conforming.side_effect
        calls = 0

        def changed(value):
            nonlocal calls
            calls += 1
            vertices, triangles = get_mesh(value)
            return vertices, triangles if calls == 1 else [0, 2, 1]

        subsystem.get_mesh_data_for_conforming.side_effect = changed
        with self.assertRaisesRegex(RuntimeError, "canonical mesh topology"):
            run(request, unreal, lambda value: None)
        self.assertEqual(state["live"], original)
        self.assertEqual(state["committed"], original)
        subsystem.remove_object_to_edit.assert_called_once()

    def test_occupied_namespace_never_duplicates_a_character(self):
        unreal, _, _, _, request = self.fixture()
        unreal.EditorAssetLibrary.list_assets.side_effect = None
        unreal.EditorAssetLibrary.list_assets.return_value = ["/Game/Existing"]
        with self.assertRaisesRegex(ValueError, "occupied"):
            run(request, unreal, lambda value: None)
        unreal.EditorAssetLibrary.duplicate_asset.assert_not_called()

    def test_invalid_steps_and_controls_are_rejected(self):
        for step in (True, 0, float("nan"), .5):
            unreal, _, _, _, request = self.fixture()
            request["step_cm"] = step
            with self.assertRaisesRegex(ValueError, "step"):
                run(request, unreal, lambda value: None)
            unreal.EditorAssetLibrary.duplicate_asset.assert_not_called()
        for indices in ([True], [-1], [3], [1, 1], []):
            unreal, subsystem, _, _, request = self.fixture()
            request["indices"] = indices
            with self.assertRaisesRegex(ValueError, "indices"):
                run(request, unreal, lambda value: None)
            subsystem.translate_face_landmarks.assert_not_called()
            subsystem.remove_object_to_edit.assert_called_once()


class NativeResponsePersistenceTests(unittest.TestCase):
    def measure(self, failure=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = Store(root / "data", "subject")
        project = root / "Project" / "Test.uproject"
        source = project.parent / "Content" / "Heads" / "Person.uasset"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"native source")
        project.write_text("{}")
        tools = Mock()
        tools.paths = {}

        def execute(name, operation, args, directory, environment):
            request = json.loads(Path(environment["AILIAS_NATIVE_RESPONSE_REQUEST"]).read_text())
            output = Path(request["output"])
            np.zeros(9, dtype="<f4").tofile(output / "reference.f32")
            values = np.zeros(27, dtype="<f4")
            if failure == "nonfinite":
                values[0] = float("nan")
            values.tofile(output / "responses.f32")
            (output / "triangles.json").write_text("[0,1,2]")
            result = {
                "status": "native_response_measured", "source_unchanged": True,
                "vertex_count": 3, "indices": [0], "responses": [{}, {}, {}],
            }
            if failure == "incomplete":
                result["responses"].pop()
            if failure == "source":
                source.write_bytes(b"changed native source")
            Path(request["receipt"]).write_text(json.dumps(result))
            return {"seconds": 1}

        tools.run.side_effect = execute
        result = measure_response(store, "/Game/Heads/Person.Person", root / "Editor.exe",
                                  project, tools, indices=[0])
        retained = store.resolve(str(Path("facefitting") / "responses" / result["revision"] / "manifest.json"))
        self.assertEqual(json.loads(retained.read_text())["status"], result["status"])
        self.assertFalse(result["native_fit_applied"])
        return result

    def test_complete_native_measurement_is_retained_without_claiming_a_fit(self):
        result = self.measure()
        self.assertEqual(result["status"], "native_response_measured_awaiting_geometry_fit")

    def test_incomplete_nonfinite_or_changed_source_results_are_not_promoted(self):
        for failure in ("incomplete", "nonfinite", "source"):
            with self.subTest(failure=failure):
                result = self.measure(failure)
                self.assertEqual(result["status"], "failed")
