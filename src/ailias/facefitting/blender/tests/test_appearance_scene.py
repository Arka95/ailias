"""Fresh Blender scene creation preserves geometry and cannot skip failed exports."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.facefitting.blender.appearance_scene import create_blender_head
from ailias.facefitting.geometry.artifact_io import export_obj
from ailias.storage import Store, atomic_json


class BlenderAppearanceTests(unittest.TestCase):
    def execute(self, *, changed_geometry=False, failure=False, hair_strands=4, requested_hair=4):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = Store(Path(temporary.name), "subject")
        store.root.mkdir()
        root = store.root / "blender"
        vertices = np.array([[0., .2, 0.], [.1, .2, 0.], [0., .3, .1]])
        triangles = np.array([[0, 1, 2]])
        model = SimpleNamespace(
            neutral=lambda identity: vertices, triangles=triangles,
            sample=lambda points: np.tile([0., .2, 0.], (478, 1)),
            groups={"skin_exterior": np.ones(3), "eye_exteriors": np.zeros(3)},
        )
        baked = {
            "color": np.ones((1, 3))*.4, "coverage": np.array([True]), "skin_color": np.ones(3)*.4,
            "hair_probability": np.ones(3), "hair_observed": np.ones(3, dtype=bool),
            "hair_color": np.ones(3)*.1, "hair_length": .02, "report": {},
        }
        mapping = (np.array([0]), np.array([0]), np.array([0]), np.ones((1, 3))/3)

        def clay(*args):
            scene = root / "head-review.blend"
            scene.write_bytes(b"same-run synthetic scene")
            return {"scene": store.reference_artifact(scene)}, {}

        def render(*args):
            if failure:
                raise RuntimeError("owned Blender process failed")
            export_obj(root / "head-for-unreal.obj",
                       vertices[:, [2, 0, 1]]*(200 if changed_geometry else 100), triangles)
            (root / "textured-head.blend").write_bytes(b"new synthetic textured scene")
            atomic_json(root / "appearance-runtime.json", {
                "geometry_maximum_deviation": 0, "cycles_device": "GPU",
                "hair": {"status": "disabled" if requested_hair == hair_strands == 0 else "approximation",
                         "strands": hair_strands},
                "hair_objects": ["hair"] if hair_strands else [], "groom": "not_exported",
            })
            for angle in (-45, 0, 45, 90):
                Image.new("RGB", (8, 8), (40, 50, 60)).save(root / f"appearance-{angle}.png")
            return {}

        with patch("ailias.facefitting.blender.appearance_scene.project_photo_evidence",
                   return_value=(baked, [])), \
                patch("ailias.facefitting.blender.appearance_scene.uv_surface_map", return_value=mapping), \
                patch("ailias.facefitting.blender.appearance_scene.render_clay_portrait", side_effect=clay), \
                patch("ailias.facefitting.blender.appearance_scene.validate_material_textures"):
            try:
                result = create_blender_head(
                    store, root, model, {"identity": np.zeros(1)}, [], {}, np.zeros((1, 3, 2)),
                    SimpleNamespace(run=render), SimpleNamespace(blender_device="CUDA", render_size=8),
                    {"texture_size": 8, "max_views": 1, "photo_size": 8, "device": "cuda:0",
                     "hair_strands": requested_hair},
                )
                return result, json.loads((root / "manifest.json").read_text())
            except RuntimeError as error:
                return error, json.loads((root / "manifest.json").read_text())

    def test_scene_and_unreal_mesh_are_built_from_the_same_run_and_retained(self):
        result, receipt = self.execute()
        self.assertEqual(result["status"], "blender_head_created")
        self.assertEqual(receipt["status"], "blender_head_created")
        self.assertFalse(result["previous_scene_reused"])
        self.assertTrue(result["geometry_preserved"])
        self.assertIn("head_only_mesh", result["artifacts"])

    def test_changed_blender_geometry_fails_instead_of_being_handed_to_unreal(self):
        error, receipt = self.execute(changed_geometry=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("coordinate basis", receipt["error"]["message"])

    def test_process_failure_is_persisted_and_not_reported_as_a_saved_scene(self):
        error, receipt = self.execute(failure=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("owned Blender process failed", receipt["error"]["message"])

    def test_missing_requested_hair_cannot_be_accepted_as_complete(self):
        error, receipt = self.execute(hair_strands=0)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("photo-supported hair", receipt["error"]["message"])

    def test_disabled_hair_saves_a_hair_free_blender_head(self):
        result, receipt = self.execute(hair_strands=0, requested_hair=0)
        self.assertEqual(result["status"], "blender_head_created")
        self.assertEqual(receipt["hair"], {"status": "disabled", "strands": 0})

    def test_unexpected_hair_fails_when_the_owner_disabled_it(self):
        error, receipt = self.execute(requested_hair=0)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("disabled hair policy", receipt["error"]["message"])
