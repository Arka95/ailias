import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ailias.facefitting.geometry.photo_camera import perspective_camera, prepare_photo_tracking
from ailias.storage import Store, atomic_json


class PhotoCameraTests(unittest.TestCase):
    def test_existing_cv_projection_is_preserved_in_world_camera_axes(self):
        parameters = [.13, -.2, .08, .1, -.2, 1.2, math.log(1.3)]
        center = np.array([.01, .25, .1])
        view = {"camera": parameters, "source_size": [800, 600]}
        camera = perspective_camera(view, center, [400, 300])
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        cv = np.array([.12, -.07, 1.5])
        world = center+rotation.T@(cv-parameters[3:6])
        ray = world-np.array(camera["position_gnm"])
        forward, up = np.array(camera["forward_gnm"]), np.array(camera["up_gnm"])
        right = np.cross(forward, up)
        depth = ray@forward
        focal = 400/(2*math.tan(math.radians(camera["horizontal_fov_degrees"])/2))
        projected = np.array([ray@right, -(ray@up)])/depth*focal+[200, 150]
        expected = cv[:2]/cv[2]*(math.exp(parameters[6])*800)*.5+[200, 150]
        np.testing.assert_allclose(projected, expected, atol=1e-10)

    def test_distorted_or_nonfinite_camera_inputs_fail_before_native_execution(self):
        view = {"camera": [0, 0, 0, 0, 0, 1, 0], "source_size": [800, 600]}
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            perspective_camera(view, [0, 0, 0], [512, 512])
        view["camera"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            perspective_camera(view, [0, 0, 0], [400, 300])


class PhotoTrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / "data", "test")
        self.source = self.root / "original.png"
        Image.new("RGB", (80, 60), (150, 100, 70)).save(self.source)
        self.alpha = self.root / "alpha.png"
        Image.new("L", (20, 30), 255).save(self.alpha)
        reference = self.store.reference_artifact(self.source)
        self.entry = {
            "face_id": "face", "source_id": reference["sha256"], "source": reference,
            "foreground_alpha": self.store.reference_artifact(self.alpha),
            "portrait_bounds": [20, 10, 40, 40],
        }
        inputs, cameras, observations = [self.root / name for name in
                                        ("inputs.json", "cameras.json", "observations.json")]
        atomic_json(inputs, {"inputs": {"entries": [self.entry]}})
        atomic_json(cameras, {
            "model_center": [0, .25, .1], "views": [
                {"face_id": "face", "source_size": [80, 60], "camera": [0, 0, 0, 0, 0, 1, 0]}]})
        atomic_json(observations, {"faces": [
            {"face_id": "face", "status": "ready", "source_id": reference["sha256"],
             "source_size": [80, 60], "blendshapes": {"mouthSmileLeft": .03}}]})
        self.fit = self.store.resolve(r"facefitting\runs\fit\manifest.json")
        self.fit.parent.mkdir(parents=True)
        atomic_json(self.fit, {
            "inputs": self.store.reference_artifact(inputs),
            "artifacts": {"cameras": self.store.reference_artifact(cameras),
                          "observations": self.store.reference_artifact(observations)},
        })
        self.output = self.root / "tracking"

    def test_actual_photo_pixels_and_full_frame_projection_are_preserved(self):
        result = prepare_photo_tracking(self.store, "fit", "face", self.output)
        with Image.open(self.output / "tracking-view.png") as image:
            self.assertEqual(image.size, (80, 60))
            self.assertEqual(image.getpixel((25, 20)), (150, 100, 70))
            self.assertEqual(image.getpixel((0, 0)), (128, 128, 128))
        self.assertFalse(result["geometry_refitted"])
        self.assertFalse(result["blender_launched"])
        self.assertTrue(all(self.store.verify(item) for item in result["input_dependencies"]))
        camera = json.loads((self.output / "tracking-camera.json").read_text())
        self.assertEqual(camera["expression"]["mouthSmileLeft"], .03)
        self.assertFalse(camera["expression_matched"])

    def test_unknown_face_does_not_create_outputs(self):
        with self.assertRaisesRegex(ValueError, "one existing"):
            prepare_photo_tracking(self.store, "fit", "other", self.output)
        self.assertFalse(self.output.exists())

    def test_changed_original_cannot_be_substituted_for_existing_camera_evidence(self):
        Image.new("RGB", (80, 60), (20, 30, 40)).save(self.source)
        with self.assertRaisesRegex(ValueError, "source evidence changed"):
            prepare_photo_tracking(self.store, "fit", "face", self.output)
        self.assertFalse(self.output.exists())

    def test_alpha_grid_must_match_source_crop_not_an_aligned_analysis_cutout(self):
        Image.new("L", (10, 10), 255).save(self.alpha)
        inputs_ref = json.loads(self.fit.read_text())["inputs"]
        inputs_path = self.store.artifact_path(inputs_ref)
        self.entry["foreground_alpha"] = self.store.reference_artifact(self.alpha)
        atomic_json(inputs_path, {"inputs": {"entries": [self.entry]}})
        fitted = json.loads(self.fit.read_text())
        fitted["inputs"] = self.store.reference_artifact(inputs_path)
        atomic_json(self.fit, fitted)
        with self.assertRaisesRegex(ValueError, "native-grid portrait alpha"):
            prepare_photo_tracking(self.store, "fit", "face", self.output)
