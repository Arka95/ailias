"""Numerical fitting tests run on CPU with CUDA telemetry mocked, never on a real GPU."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ailias.head_reconstruction.settings import FitPolicy
from ailias.head_reconstruction.geometry.cuda_optimization import fit_cuda
from ailias.head_reconstruction.geometry.optimization import fit, project


class BatchedOptimizationTests(unittest.TestCase):
    def fixture(self):
        random = np.random.default_rng(42)
        points = random.uniform([-.07, -.09, -.04], [.07, .09, .04], (120, 3))
        identity = np.zeros((1, 120, 3))
        identity[0, :, 0] = .012*np.cos(points[:, 1]*25)
        expression = np.zeros((1, 120, 3))
        expression[0, :, 1] = .01*(points[:, 1] < -.025)
        model = SimpleNamespace(points=points, identity_points=identity,
                                expression_points=expression, center=np.zeros(3))
        cameras = np.array([[3.1, yaw, 0, 0, 0, .7, 0] for yaw in (-.4, 0., .4)])
        targets, _ = project(points[None]+.8*identity[0], cameras, model.center)
        views = [{"face_id": str(i), "target": targets[i], "camera": cameras[i],
                  "eye_span": .15, "weights": np.full(120, 1/120), "view_weight": 1,
                  "source_size": np.array([1000, 1000])} for i in range(3)]
        return model, views

    def test_jacobian_batching_keeps_every_view_and_matches_full_batch_shape(self):
        import torch

        model, views = self.fixture()
        policy = FitPolicy(device="cpu", identity_candidates=(1,), expression_modes=(1, 1, 1),
                           identity_regularization=1e-8, expression_regularization=1e-8,
                           max_iterations=80, view_batch_size=1)
        with patch("ailias.head_reconstruction.geometry.cuda_optimization.load_torch", return_value=torch), \
                patch.multiple(torch.cuda, synchronize=lambda device: None,
                               get_device_name=lambda device: "unit-cpu",
                               max_memory_allocated=lambda device: 0):
            small = fit_cuda(model, views, policy, 1)
            full = fit_cuda(model, views, replace(policy, view_batch_size=8), 1)
            fixed = fit_cuda(model, views, policy, 1, fixed_identity=small["identity"])
        self.assertEqual(len(small["report"]["views"]), len(views))
        self.assertEqual(small["report"]["batch_views"], 1)
        self.assertAlmostEqual(small["identity"][0], .8, delta=.15)
        np.testing.assert_allclose(small["identity"], full["identity"], atol=.01)
        self.assertLess(fixed["report"]["median_outer_eye_span_error"], .005)

    def test_wide_angle_focal_length_is_not_forced_above_the_old_floor(self):
        model, views = self.fixture()
        model.identity_points = np.zeros((0, 120, 3))
        model.expression_points = np.zeros((0, 120, 3))
        true_camera = np.array([[3.1, .2, 0, 0, 0, .25, np.log(.25)]])
        target, _ = project(model.points[None], true_camera, model.center)
        view = {**views[0], "target": target[0],
                "camera": np.array([3.1, .2, 0, 0, 0, .7, np.log(.7)])}
        result = fit(model, [view], FitPolicy(max_nfev=100), 0, fixed_identity=np.zeros(0))
        self.assertLess(result["report"]["views"][0]["focal_factor"], .4)
        self.assertLess(result["report"]["median_outer_eye_span_error"], .005)
