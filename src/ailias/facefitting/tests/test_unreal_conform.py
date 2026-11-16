"""Test local Character solve/commit contracts with a fake editor and no native/model execution."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from ailias.facefitting.unreal_conform import conform_head
from ailias.facefitting.unreal_conform_worker import run
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
            def get_path_name(self):
                return "/Game/Test/HeadOnlyDraft.HeadOnlyDraft"

        character = Character()
        point = SimpleNamespace(x=1, y=2, z=3)
        subsystem = Mock()
        subsystem.get_mesh_data_for_conforming.return_value = ([point]*3, [0, 1, 2])
        subsystem.track_face_landmarks_from_image.return_value = {
            "nose": SimpleNamespace(tracking_points=[SimpleNamespace(x=.5, y=.5)])}
        subsystem.try_add_object_to_edit.return_value = True
        subsystem.conform_to_target_meshes.return_value = True
        subsystem.get_face_landmarks.side_effect = [[point], [SimpleNamespace(x=2, y=3, z=4)]]
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
        subsystem.remove_object_to_edit.assert_called_once()
        subsystem.request_auto_rigging.assert_not_called()
        subsystem.request_texture_sources.assert_not_called()

    def test_native_failure_never_commits_or_saves_and_releases_editor_state(self):
        unreal, subsystem, _, request = self.fixture()
        subsystem.conform_to_target_meshes.return_value = False
        with self.assertRaisesRegex(RuntimeError, "returned false"):
            run(request, unreal, lambda value: None)
        subsystem.commit_face_state.assert_not_called()
        unreal.EditorAssetLibrary.save_loaded_asset.assert_not_called()
        subsystem.remove_object_to_edit.assert_called_once()

    def test_failed_tracking_does_not_create_or_register_a_character(self):
        unreal, subsystem, tools, request = self.fixture()
        subsystem.track_face_landmarks_from_image.return_value = None
        with self.assertRaisesRegex(RuntimeError, "tracking failed"):
            run(request, unreal, lambda value: None)
        tools.create_asset.assert_not_called()
        subsystem.try_add_object_to_edit.assert_not_called()


class ConformPersistenceTests(unittest.TestCase):
    def conform(self, outcome):
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
        project = root / "Project" / "Review.uproject"
        project.parent.mkdir()
        project.write_text("{}")
        imported = {"status": "static_head_imported_for_review", "revision": "appearance",
                    "source": store.reference_artifact(source), "artifacts": {},
                    "readiness": {"project": str(project)},
                    "imported": {"mesh_geometry": [{"asset": "/Game/Head.Head"}]}}
        atomic_json(imported_path, imported)
        probe = {"blockers": [], "editor": str(root / "Editor.exe"), "project": str(project)}

        def launch(name, operation, args, directory, environment=None):
            if name == "blender":
                Image.new("RGBA", (1, 1)).save(directory / "tracking-view.png")
                atomic_json(directory / "tracking-camera.json", {})
            else:
                request = json.loads(Path(environment["AILIAS_CONFORM_REQUEST"]).read_text())
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

        appearance = {"integrity_problems": [], "artifacts": {"scene": store.reference_artifact(scene)}}
        with patch("ailias.facefitting.unreal_conform.inspect_appearance", return_value=appearance), \
                patch("ailias.facefitting.unreal_conform.readiness", return_value=probe):
            result = conform_head(store, "imported", Path(probe["editor"]), project,
                                  SimpleNamespace(blender_device="CUDA"),
                                  SimpleNamespace(paths={}, run=launch))
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

    def test_native_success_without_package_is_failure_and_cleans_temporary_pixels(self):
        _, result = self.conform("missing_package")
        self.assertEqual(result["status"], "failed")

    def test_malformed_native_receipt_retains_failure_and_does_not_mask_cleanup(self):
        _, result = self.conform("malformed")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["native_receipt_unreadable"])


if __name__ == "__main__":
    unittest.main()
