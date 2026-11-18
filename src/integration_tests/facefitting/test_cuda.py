"""Explicit opt-in numerical checks; excluded from application unit discovery."""

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace

if os.environ.get("AILIAS_CUDA_FITTING_TESTS") != "1":
    raise unittest.SkipTest("Set AILIAS_CUDA_FITTING_TESTS=1 for authorized real CUDA checks")

import numpy as np
import torch

from ailias.facefitting.geometry.cuda_optimization import project as cuda_project
from ailias.facefitting.geometry.optimization import fit, project
from ailias.facefitting.configuration import FitPolicy


class CudaNumericsTests(unittest.TestCase):
    def test_projection_matches_cpu_and_has_correct_gradients(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(-.05, .05, (2, 4, 3))
        cameras = np.array([[0, 0, 0, 0, 0, .7, .1],
                            [3.1, .2, -.1, .02, 0, .8, -.1]])
        expected, _ = project(points, cameras, np.zeros(3))
        x = torch.tensor(points, dtype=torch.float64, device="cuda:0", requires_grad=True)
        camera = torch.tensor(cameras, dtype=torch.float64, device="cuda:0", requires_grad=True)
        center = torch.zeros(3, dtype=torch.float64, device="cuda:0")
        observed, _ = cuda_project(torch, x, camera, center)
        np.testing.assert_allclose(observed.detach().cpu().numpy(), expected, atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b: cuda_project(torch, a, b, center)[0], (x, camera),
            eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_cuda_recovers_shared_identity_and_opposing_expressions(self):
        rng = np.random.default_rng(42)
        points = rng.uniform([-.07, -.09, -.02], [.07, .09, .03], (120, 3))
        basis = np.zeros((1, 120, 3))
        basis[0, :, 0] = .012*np.cos(points[:, 1]*25)
        expression = np.zeros((1, 120, 3))
        expression[0, :, 1] = .01*(points[:, 1] < -.025)
        model = SimpleNamespace(points=points, identity_points=basis,
                                expression_points=expression, center=np.zeros(3))
        cameras = np.array([[3.1, 0, 0, 0, 0, .7, 0], [3.1, .4, 0, .02, 0, .65, .1],
                            [3.1, -.4, 0, -.02, 0, .75, -.1]])
        expressions = np.array([[-.7], [.8], [.1]])
        world = points[None] + .9*basis[0] + np.einsum("ni,ivc->nvc", expressions, expression)
        target, _ = project(world, cameras, model.center)
        views = [{"face_id": str(i), "target": target[i], "camera": cameras[i],
                  "eye_span": .15, "weights": np.full(120, 1/120), "view_weight": 1,
                  "source_size": np.array([1000, 1000])} for i in range(3)]
        policy = FitPolicy(device="cuda:0", identity_candidates=(1,), expression_modes=(1, 1, 1),
                           identity_regularization=1e-8, expression_regularization=1e-8)
        fitted = fit(model, views, policy, 1)
        reference = fit(model, views, replace(policy, device="cpu"), 1)
        self.assertAlmostEqual(fitted["identity"][0], .9, delta=.15)
        self.assertLess(fitted["per_view"][0, 7], -.3)
        self.assertGreater(fitted["per_view"][1, 7], .3)
        self.assertLess(fitted["report"]["objective"], max(reference["report"]["objective"]*10, 1e-6))
        self.assertEqual(fitted["report"]["autograd_device"], "cuda:0")
        print("CUDA_SYNTHETIC_PROOF", fitted["report"])


if __name__ == "__main__":
    unittest.main()
