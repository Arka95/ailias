"""Opt-in real CUDA proof for expression-aware UV photo sampling."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


@unittest.skipUnless(os.environ.get("AILIAS_CUDA_FITTING_TESTS") == "1", "Explicit CUDA opt-in required")
class AppearanceCudaTests(unittest.TestCase):
    def test_constant_photo_reaches_visible_uv_surface_in_correct_source_coordinates(self):
        from ailias.facefitting.appearance.uv_rasterization import uv_surface_map
        from ailias.facefitting.appearance.cuda_texture_projection import bake

        vertices = np.array([[-.2, -.2, 0], [.2, -.2, 0], [-.2, .2, 0]])
        triangles = np.array([[0, 2, 1]])
        model = SimpleNamespace(vertices=vertices, triangles=triangles,
                                expression=np.zeros((1, 3, 3)), center=np.zeros(3),
                                neutral=lambda identity: vertices)
        mapping = uv_surface_map(np.array([[[0., 0.], [0., 1.], [1., 0.]]]), 16)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "photo.npz"
            rgb = np.broadcast_to(np.array([128, 64, 32], dtype=np.uint8), (32, 32, 3)).copy()
            np.savez(path, rgb=rgb, alpha=np.full((32, 32), 255, np.uint8),
                     labels=np.ones((32, 32), np.uint8))
            views = [{"face_id": "test", "camera": [0, 0, 0, 0, 0, 1, 0],
                      "expression": [0], "source_size": [40, 40], "appearance_weight": 1,
                      "view": {"yaw_deg": 0}}]
            photos = [{"data": str(path), "skin_color": (np.array([128, 64, 32])/255).tolist(),
                       "portrait_scale": [1, 1], "hair_color": None}]
            result = bake(model, np.zeros(0), mapping, views,
                          {"test": {"portrait_bounds": [4, 4, 36, 36]}}, photos, "cuda:0")
            self.assertGreater(result["coverage"].mean(), .9)
            np.testing.assert_allclose(result["color"][result["coverage"]],
                                       np.broadcast_to([128/255, 64/255, 32/255],
                                                       result["color"][result["coverage"]].shape),
                                       atol=1e-5)


if __name__ == "__main__":
    unittest.main()
