import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ailias.head_reconstruction.geometry.optimization import (
    evaluation_score,
    fit,
    initialize_camera,
    project,
    validation_indices,
)
from ailias.head_reconstruction.geometry.head_model import correspondence, sample_vertices
from ailias.head_reconstruction.settings import FitPolicy


class CorrespondenceTests(unittest.TestCase):
    def test_interior_contributions_are_excluded_but_zero_weight_corners_are_allowed(self):
        vertices = np.zeros((4, 3))
        triangles = np.array([[0, 1, 2], [1, 2, 3]])
        table = [[0, .5, 1, .5, 2, 0], [1, .5, 2, .5, 3, 0]]
        indices, weights, supported = correspondence(table, vertices, triangles, np.array([1, 1, 0, 1]))
        self.assertEqual(supported.tolist(), [True, False])
        points = sample_vertices(np.arange(12).reshape(4, 3), indices, weights)
        np.testing.assert_allclose(points[0], [1.5, 2.5, 3.5])

    def test_bad_topology_or_weights_are_rejected(self):
        vertices = np.zeros((4, 3))
        for table in ([[0, .5, 1, .4, 2, 0]], [[0, .5, 1, .5, 3, 0]],
                      [[0.1, .5, 1, .5, 2, 0]]):
            with self.subTest(table=table), self.assertRaises(ValueError):
                correspondence(table, vertices, np.array([[0, 1, 2]]), np.ones(4))


class FitTests(unittest.TestCase):
    def test_camera_initialization_recovers_when_sqpnp_rejects_a_focal_trial(self):
        import cv2

        rng = np.random.default_rng(42)
        points = rng.uniform([-.07, -.09, -.02], [.07, .09, .03], (120, 3))
        camera = np.array([3.1, 0, 0, 0, 0, .7, np.log(1.2)])
        target, _ = project(points[None], camera[None], np.zeros(3))
        solve = cv2.solvePnP
        attempted = []

        def rejecting_sqpnp(*args, **kwargs):
            attempted.append(kwargs["flags"])
            if kwargs["flags"] == cv2.SOLVEPNP_SQPNP:
                raise cv2.error("degenerate SQPNP coordinate variance")
            return solve(*args, **kwargs)

        with patch.object(cv2, "solvePnP", side_effect=rejecting_sqpnp):
            fitted = initialize_camera(points, target[0], np.zeros(3),
                                       np.ones(120, dtype=bool), .15)
        prediction, depth = project(points[None], fitted[None], np.zeros(3))
        self.assertIn(cv2.SOLVEPNP_EPNP, attempted)
        self.assertTrue(np.all(depth > 0))
        np.testing.assert_allclose(prediction, target, atol=1e-5)

    def test_camera_solver_failures_become_an_accountable_view_rejection(self):
        import cv2

        with patch.object(cv2, "solvePnP", side_effect=cv2.error("degenerate coordinates")):
            with self.assertRaisesRegex(ValueError, "Could not initialize"):
                initialize_camera(np.zeros((120, 3)), np.zeros((120, 2)), np.zeros(3),
                                  np.ones(120, dtype=bool), .15)

    def test_requested_cuda_never_silently_falls_back_to_cpu(self):
        from ailias.head_reconstruction.geometry.cuda_optimization import load_torch

        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake}), self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
            load_torch("cuda:0")

    def test_shared_shape_is_recovered_without_baking_per_view_expression_into_it(self):
        rng = np.random.default_rng(42)
        points = rng.uniform([-.07, -.09, -.02], [.07, .09, .03], (120, 3))
        basis = np.zeros((1, 120, 3))
        basis[0, :, 0] = .012 * np.cos(points[:, 1]*25)
        expression = np.zeros((1, 120, 3))
        expression[0, :, 1] = .01 * (points[:, 1] < -.025)
        model = SimpleNamespace(points=points, identity_points=basis,
                                expression_points=expression, center=np.zeros(3))
        cameras = np.array([[3.1, 0, 0, 0, 0, .7, 0],
                            [3.1, .4, 0, .02, 0, .65, .1],
                            [3.1, -.4, 0, -.02, 0, .75, -.1]])
        expressions = np.array([[-.7], [.8], [.1]])
        world = points[None] + .9*basis[0] + np.einsum("ni,ivc->nvc", expressions, expression)
        targets, _ = project(world, cameras, model.center)
        views = [{"face_id": str(i), "target": targets[i], "camera": cameras[i],
                  "eye_span": .15, "weights": np.full(120, 1/120), "view_weight": 1,
                  "source_size": np.array([1000, 1000])} for i in range(3)]
        policy = FitPolicy(identity_candidates=(1,), expression_modes=(1, 1, 1),
                           identity_regularization=1e-8, expression_regularization=1e-8,
                           max_nfev=45)
        baseline = fit(model, views, policy, 1, fixed_identity=np.zeros(1))
        fitted = fit(model, views, policy, 1)
        self.assertLess(fitted["report"]["median_outer_eye_span_error"],
                        baseline["report"]["median_outer_eye_span_error"] * .3)
        self.assertAlmostEqual(fitted["identity"][0], .9, delta=.15)
        self.assertLess(fitted["per_view"][0, 7], -.3)
        self.assertGreater(fitted["per_view"][1, 7], .3)

    def test_split_holds_out_views_without_removing_singleton_angle(self):
        views = [{"face_id": str(i), "view": {"category": "front"}} for i in range(9)]
        views.append({"face_id": "side", "view": {"category": "profile"}})
        heldout = validation_indices(views)
        self.assertEqual(heldout, [4])
        self.assertNotIn(9, heldout)

    def test_candidate_evaluation_does_not_hide_side_views_behind_many_frontals(self):
        views = [{"face_id": str(i), "view": {"category": "front" if i < 10 else "oblique"}}
                 for i in range(11)]
        report = {"views": [{"face_id": str(i), "median_outer_eye_span_error": .01 if i < 10 else .09}
                            for i in range(11)], "median_outer_eye_span_error": .01}
        score = evaluation_score(report, views)
        self.assertAlmostEqual(score["view_balanced_mean_error"], .05)
        self.assertEqual(score["median_view_error"], .01)


if __name__ == "__main__":
    unittest.main()
