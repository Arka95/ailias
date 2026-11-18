import math
import unittest

import numpy as np

from ailias.facefitting.geometry.optimization import project
from ailias.facefitting.geometry.sculpt_fitting import (
    corrective_step,
    fit_photo_pose,
    proposed_vertices,
)


class NativePhotoFitTests(unittest.TestCase):
    def test_native_response_correction_reduces_the_observed_photo_shape_error(self):
        points = np.array([[-.1, -.1, 0], [.1, -.1, 0], [-.1, .1, 0], [.1, .1, 0.]])
        basis = np.zeros((3, 4, 3))
        basis[0, :, 0] = np.sign(points[:, 0])*.01
        camera = np.array([0., 0, 0, 0, 0, 1, 0])
        expected = points+basis[0]*.4
        target, _ = project(expected[None], camera[None], np.zeros(3))
        view = {"camera": camera, "center": np.zeros(3), "target": target[0],
                "supported": np.ones(4, dtype=bool), "eye_span": .2}
        values, report = corrective_step(points, basis, [view])
        actual, _ = project((points+np.einsum("k,kpc->pc", values, basis))[None],
                            camera[None], np.zeros(3))
        before, _ = project(points[None], camera[None], np.zeros(3))
        self.assertLess(np.linalg.norm(actual-target), np.linalg.norm(before-target)*.05)
        self.assertLessEqual(np.linalg.norm(values), .5)
        self.assertFalse(report["native_edits_applied"])
        self.assertFalse(report["likeness_verified"])

    def test_whole_mesh_displacement_budget_scales_the_actual_proposal(self):
        reference = np.zeros((2, 3))
        responses = np.zeros((3, 2, 3))
        responses[0, :, 0] = 2
        target, values = proposed_vertices(reference, responses, [.5, 0, 0], .2)
        np.testing.assert_allclose(target[:, 0], [.2, .2])
        np.testing.assert_allclose(values, [.1, 0, 0])

    def test_pose_fit_preserves_focal_and_does_not_change_identity_geometry(self):
        points = np.random.default_rng(7).uniform([-.1, -.1, 0], [.1, .1, .05], (468, 3))
        points -= points.mean(0)
        original = points.copy()
        camera = np.array([.05, .1, -.02, .02, -.01, .8, math.log(1.2)])
        projected, _ = project(points[None], camera[None], np.zeros(3))
        pixels = projected[0]*1000+[500, 400]
        observation = {
            "face_id": "synthetic", "source_size": [1000, 800], "source_xy": pixels.tolist(),
            "on_head": [True]*468, "eye_span_source": float(np.linalg.norm(pixels[33]-pixels[263])),
        }
        result = fit_photo_pose(points, observation, np.ones(468, dtype=bool), 1.2)
        actual, _ = project(points[None], result["camera"][None], result["center"])
        np.testing.assert_allclose(actual, projected, atol=1e-6)
        self.assertAlmostEqual(math.exp(result["camera"][6]), 1.2)
        np.testing.assert_array_equal(points, original)
        self.assertFalse(result["focal_calibrated"])

    def test_invalid_budget_or_empty_photo_support_cannot_produce_a_target(self):
        with self.assertRaises(ValueError):
            proposed_vertices(np.zeros((2, 3)), np.zeros((3, 2, 3)), [1, 0, 0], 0)
        with self.assertRaises(ValueError):
            corrective_step(np.zeros((4, 3)), np.zeros((3, 4, 3)), [])
