"""Pipeline boundary failures remain persistent, specific and non-successful."""

import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.facefitting.workflows.photo_to_character import run_photo_to_character_pipeline
from ailias.facefitting.geometry.artifact_io import export_obj
from ailias.facefitting.unreal.character_features import character_package
from ailias.storage import Store, atomic_json


class PhotoToCharacterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name), "subject")
        self.store.root.mkdir()
        image = self.store.resolve("retained.png")
        Image.new("RGB", (16, 12)).save(image)
        self.manifest = self.store.resolve("sampled.json")
        self.reference = self.store.reference_artifact(image)
        atomic_json(self.manifest, {"status": "video_evidence_ready", "frames": [{
            "face_id": "frame", "image": self.reference, "source_id": "video",
        }]})
        self.policy = SimpleNamespace(record=lambda: {})

    def invoke(self):
        return run_photo_to_character_pipeline(
            self.store, self.manifest, Path("Editor.exe"),
            Path("Project.uproject"), SimpleNamespace(), self.policy,
        )

    def persisted(self, result):
        return json.loads(self.store.resolve(
            "facefitting\\fresh-video\\" + result["revision"] + "\\manifest.json").read_text())

    def test_blocked_preflight_is_a_failed_stage_and_never_starts_inference_or_decoding(self):
        with patch("ailias.facefitting.workflows.photo_to_character.readiness", return_value={"blockers": ["missing_editor"]}), \
                patch("ailias.facefitting.workflows.photo_to_character.extract_photo_evidence") as process:
            result = self.invoke()
        self.assertEqual(self.persisted(result)["status"], "failed")
        self.assertEqual(result["stages"]["native_preflight"]["status"], "failed")
        self.assertIn("missing_editor", result["error"]["message"])
        self.assertFalse(result["video_decoding_performed"])
        process.assert_not_called()

    def test_photo_stage_exception_preserves_the_root_cause_and_stops_before_fitting(self):
        try:
            raise OSError("model read failed")
        except OSError as cause:
            error = RuntimeError("photo failed")
            error.__cause__ = cause
        with patch("ailias.facefitting.workflows.photo_to_character.readiness", return_value={"blockers": []}), \
                patch("ailias.facefitting.workflows.photo_to_character.model_assets", return_value={
                    "landmarker": {"artifact": self.reference}}), \
                patch("ailias.facefitting.workflows.photo_to_character.extract_photo_evidence", side_effect=error), \
                patch("ailias.facefitting.workflows.photo_to_character.fit_multiview_identity") as fit:
            result = self.invoke()
        saved = self.persisted(result)
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["current_stage"], "fresh_photo_processing")
        self.assertEqual(saved["error"]["cause"], "model read failed")
        self.assertEqual(saved["stages"]["fresh_photo_processing"]["status"], "failed")
        self.assertTrue(self.store.verify(self.reference))
        fit.assert_not_called()

    def complete(self, *, appearance_fails=False, blender_fails=False):
        project = self.store.resolve("Project\\Preview.uproject")
        project.parent.mkdir()
        project.write_text("{}")
        archive = self.store.resolve("generic-model.npz")
        np.savez(archive, triangle_uvs=np.array([[[0., 0.], [1., 0.], [0., 1.]]]))
        assets = {name: {"artifact": self.store.reference_artifact(archive)}
                  for name in ("gnm", "correspondence", "landmarker")}
        colour = {"status": "observed", "median_linear_rgb": [.1, .05, .025]}
        region = {"status": "estimated", "dark_fraction": .4, "presence_estimate": True,
                  "median_dark_linear_rgb": [.02, .015, .01]}
        frame = {"face_id": "frame", "image": self.reference, "source_size": [16, 12],
                 "source_xy": [[1., 1.]], "quality_score": 1}
        evidence = {"frames": [frame], "parser": {}, "faces": {"frame": {
            "status": "observed", "features": {"iris": colour,
                "regions": {"scalp_hair": colour, "eyebrows": colour},
                "facial_hair_appearance": {"regions": {"beard": region, "moustache": region}}}}}}
        vertices = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, .1]])
        model = SimpleNamespace(vertices=vertices, neutral=lambda identity: vertices,
                                triangles=np.array([[0, 1, 2]]), center=np.zeros(3))
        fitted = {"identity": np.zeros(1), "per_view": np.zeros((1, 8))}
        source = "/Game/Fresh/Character.Character"
        candidate = "/Game/CurrentAppearance/Character.Character"

        def asset(path):
            package = character_package(project, path)
            package.parent.mkdir(parents=True, exist_ok=True)
            package.write_bytes(b"new native asset")
            return self.store.reference_artifact(package)

        operations = []

        def blender(store, root, *args):
            operations.append("blender")
            if blender_fails:
                raise RuntimeError("Blender export failed")
            root.mkdir()
            mesh = root / "head-for-unreal.obj"
            export_obj(mesh, vertices[:, [2, 0, 1]]*100, model.triangles)
            return {"status": "blender_head_created", "artifacts": {
                "head_only_mesh": store.reference_artifact(mesh)}}

        def process(store, samples, root, *args):
            output = root / "photo-evidence"
            output.mkdir()
            atomic_json(output / "manifest.json", evidence)
            return evidence

        def native(tools, project, worker, root, operation, parameters, report):
            operations.append(operation)
            if operation == "create":
                self.assertEqual(Path(parameters["obj"]).parent.name, "blender")
                self.assertTrue(Path(parameters["obj"]).is_file())
                asset(source)
                return {"status": "head_conformed_awaiting_review", "character": source,
                        "face_landmarks_saved": vertices.tolist(),
                        "face_landmarks_solved_in_target_pose": vertices.tolist(),
                        "face_landmarks_after": (vertices+[0, 0, 100]).tolist()}
            if operation == "verify":
                return {"status": "fresh_character_reloaded", "native_skin_uvs": {}}
            if operation == "import_texture":
                asset("/Game/Current/PhotoBaseColor")
                return {"status": "native_photo_texture_imported", "texture": "/Game/Current/PhotoBaseColor"}
            image = Image.new("RGB", (540, 960), (40, 40, 40))
            image.paste((180, 130, 100), (100, 100, 400, 800))
            image.save(parameters["output"])
            return {"status": "native_editor_preview_rendered"}

        def bake(store, root, *args, **kwargs):
            image = root / "native-photo-basecolor.png"
            Image.new("RGB", (32, 32), (100, 80, 60)).save(image)
            return {"texture": str(image), "hair_length_model_units": .02}

        def apply(*args, **kwargs):
            self.assertFalse(kwargs["use_working_baseline"])
            return {"status": "failed", "error": "groom missing"} if appearance_fails else {
                "status": "features_reloaded_awaiting_visual_review", "character": candidate,
                "artifacts": {"character": asset(candidate)}}

        with ExitStack() as stack:
            for name, value in {
                "readiness": lambda *args: {"blockers": []}, "model_assets": lambda *args: assets,
                "extract_photo_evidence": process, "HeadModel": lambda *args: model,
                "prepare_views": lambda *args: ([{"face_id": "frame", "view": {"category": "front"}}], []),
                "fit_multiview_identity": lambda *args: fitted, "save_fit": lambda *args: None,
                "create_blender_head": blender,
                "execute_unreal_worker": native, "bake_photo_skin_texture": bake, "apply_character_features": apply,
            }.items():
                stack.enter_context(patch("ailias.facefitting.workflows.photo_to_character." + name, side_effect=value))
            result = run_photo_to_character_pipeline(
                self.store, self.manifest, Path("Editor.exe"), project, SimpleNamespace(paths={}),
                SimpleNamespace(record=lambda: {}, expression_modes=[]),
                generation_policy=SimpleNamespace(), appearance_policy={
                    "device": "cuda:0", "texture_size": 256, "max_views": 1, "photo_size": 256})
        return result, operations

    def test_one_invocation_reaches_fresh_texture_grooms_and_a_new_preview_without_old_baseline(self):
        result, operations = self.complete()
        self.assertEqual(self.persisted(result)["status"], "fresh_video_completed")
        self.assertEqual(operations, ["blender", "create", "verify", "import_texture", "preview"])
        self.assertTrue(result["blender_used"])
        self.assertFalse(result["assembled"])
        self.assertFalse(result["likeness_verified"])
        self.assertTrue(self.store.verify(self.reference))

    def test_native_appearance_failure_cannot_skip_to_preview_or_success(self):
        result, operations = self.complete(appearance_fails=True)
        self.assertEqual(self.persisted(result)["status"], "failed")
        self.assertEqual(result["current_stage"], "apply_and_reload_native_appearance")
        self.assertIn("groom missing", result["error"]["message"])
        self.assertNotIn("preview", operations)

    def test_blender_failure_cannot_be_skipped_or_launch_unreal_creation(self):
        result, operations = self.complete(blender_fails=True)
        self.assertEqual(self.persisted(result)["status"], "failed")
        self.assertEqual(result["current_stage"], "create_blender_appearance")
        self.assertEqual(operations, ["blender"])
        self.assertIn("Blender export failed", result["error"]["message"])
