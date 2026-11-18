"""Fit geometry exports retain UV seams and reject invalid topology."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from ailias.facefitting.geometry.artifact_io import export_obj


class ObjExportTests(unittest.TestCase):
    def test_native_obj_contains_real_per_corner_uv_indices_including_seams(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]])
        triangles = np.array([[0, 1, 2], [2, 1, 0]])
        uvs = np.array([[[0., 0], [1., 0], [0., 1]], [[.2, 1.], [1., .2], [.1, .1]]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "head.obj"
            export_obj(path, vertices, triangles, triangle_uvs=uvs)
            lines = path.read_text().splitlines()
        self.assertEqual(len([s for s in lines if s.startswith("vt ")]), 6)
        self.assertEqual([s for s in lines if s.startswith("f ")],
                         ["f 1/1 2/2 3/3", "f 3/4 2/5 1/6"])
        actual = np.array([[float(v) for v in s.split()[1:]] for s in lines if s.startswith("vt ")])
        np.testing.assert_allclose(actual, uvs.reshape(-1, 2))

    def test_invalid_uv_topology_is_rejected_before_writing_a_native_obj(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "head.obj"
            with self.assertRaisesRegex(ValueError, "per-corner topology"):
                export_obj(path, np.zeros((3, 3)), np.array([[0, 1, 2]]),
                           triangle_uvs=np.zeros((1, 2, 2)))
            self.assertFalse(path.exists())
