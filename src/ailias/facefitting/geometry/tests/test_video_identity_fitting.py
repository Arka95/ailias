"""Verify coordinate and category invariants without real models or an engine."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ailias.facefitting.geometry.video_identity_fitting import (
    align_expression_basis,
    geometry_score,
    surface_quality,
)


class NativeVideoFitTests(unittest.TestCase):
    def test_expression_alignment_transforms_displacements_without_baking_translation(self):
        rng = np.random.default_rng(4)
        points = rng.normal(size=(468, 3))
        expressions = rng.normal(size=(3, 468, 3))*.01
        original = points.copy()
        matrix = Rotation.from_euler("xyz", [.2, -.3, .1]).as_matrix()
        target = points @ matrix*1.2 + [1, 2, 3]
        result, record = align_expression_basis(points, target, expressions, np.ones(468, dtype=bool))
        np.testing.assert_allclose(result, expressions @ matrix*1.2, atol=1e-12)
        np.testing.assert_array_equal(points, original)
        self.assertAlmostEqual(record["scale"], 1.2)
        self.assertFalse(record["calibrated"])

    def test_failed_or_insufficient_expression_alignment_is_not_used(self):
        with self.assertRaises(ValueError):
            align_expression_basis(np.zeros((468, 3)), np.zeros((468, 3)),
                                   np.zeros((2, 468, 3)), np.zeros(468, dtype=bool))
        with self.assertRaises(ValueError):
            align_expression_basis(np.zeros((468, 3)), np.zeros((468, 3)),
                                   np.zeros((2, 468, 3)), np.ones(468, dtype=bool))

    def test_validation_category_cannot_change_when_candidate_camera_moves(self):
        result = {"report": {
            "median_outer_eye_span_error": .2,
            "views": [{"face_id": "front", "median_outer_eye_span_error": .1,
                       "view": {"category": "profile"}},
                      {"face_id": "side", "median_outer_eye_span_error": .3,
                       "view": {"category": "front"}}]}}
        views = [{"face_id": "front", "view": {"category": "front"}},
                 {"face_id": "side", "view": {"category": "profile"}}]
        self.assertEqual(geometry_score(result, views), {"front": .1, "profile": .3})

    def test_triangle_flip_or_collapse_is_rejected_despite_small_displacement(self):
        source = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]])
        triangles = np.array([[0, 1, 2]])
        self.assertTrue(surface_quality(source, source*1.1, triangles)["passed"])
        self.assertFalse(surface_quality(source, source[[0, 2, 1]], triangles)["passed"])
        self.assertFalse(surface_quality(source, np.zeros_like(source), triangles)["passed"])
