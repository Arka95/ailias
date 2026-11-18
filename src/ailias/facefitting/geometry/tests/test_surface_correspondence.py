import unittest

import numpy as np

from ailias.facefitting.geometry.surface_correspondence import (
    project_native, ray_correspondences, rebind_native_faces, sample_native,
)


class NativeCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[-1., 0, -1], [1., 0, -1], [1., 0, 1], [-1., 0, 1]])
        self.triangles = np.array([[0, 2, 1], [0, 3, 2]])
        self.camera = {
            "position": [0, 2, 0], "forward": [0, -1, 0], "right": [1, 0, 0],
            "up": [0, 0, 1], "image_size": [100, 100], "focal_pixels": 50,
        }

    def test_exact_native_surface_samples_reproject_to_the_rendered_landmarks(self):
        expected = np.array([[-.2, 0, .4], [.3, 0, -.5]])
        pixels = project_native(expected, self.camera)
        result = ray_correspondences(self.vertices, self.triangles, pixels, self.camera)
        sampled, indices = sample_native(self.vertices, self.triangles, result)
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_allclose(sampled, expected, atol=1e-12)
        np.testing.assert_allclose(project_native(sampled, self.camera), pixels, atol=1e-12)
        self.assertFalse(result["semantic_accuracy_verified"])

    def test_nearest_visible_surface_is_used_not_a_hidden_layer(self):
        vertices = np.r_[self.vertices, self.vertices+[0, 1, 0]]
        topology = np.r_[self.triangles, self.triangles+4]
        result = ray_correspondences(vertices, topology, [[50, 50]], self.camera)
        sampled, _ = sample_native(vertices, topology, result)
        self.assertAlmostEqual(sampled[0, 1], 1)

    def test_misses_and_backfaces_abstain_instead_of_using_a_nearest_vertex(self):
        result = ray_correspondences(self.vertices, self.triangles, [[0, 0], [1000, 50]], self.camera)
        np.testing.assert_array_equal(result["supported"], [False, False])
        sampled, indices = sample_native(self.vertices, self.triangles, result)
        self.assertEqual(len(sampled), 0)
        self.assertEqual(len(indices), 0)
        reversed_faces = self.triangles[:, ::-1]
        result = ray_correspondences(self.vertices, reversed_faces, [[50, 50]], self.camera)
        self.assertFalse(result["supported"][0])

    def test_invalid_camera_and_topology_do_not_produce_correspondences(self):
        for field, value in (("up", [1, 0, 0]), ("focal_pixels", 0), ("position", [float("nan"), 0, 0])):
            camera = {**self.camera, field: value}
            with self.assertRaises(ValueError):
                ray_correspondences(self.vertices, self.triangles, [[50, 50]], camera)
        with self.assertRaises(ValueError):
            ray_correspondences(self.vertices, [[0, 2, 10]], [[50, 50]], self.camera)

    def test_exact_native_face_permutations_preserve_sampled_position(self):
        points = np.array([[-.2, 0, .4], [.3, 0, -.5]])
        original = ray_correspondences(
            self.vertices, self.triangles, project_native(points, self.camera), self.camera)
        target = self.triangles[::-1, ::-1]
        rebound = rebind_native_faces(self.triangles, target, original)
        sampled, _ = sample_native(self.vertices, target, rebound)
        np.testing.assert_allclose(sampled, points, atol=1e-12)
        self.assertEqual(rebound["unmatched_landmark_indices"], [])

    def test_retriangulated_faces_abstain_rather_than_assuming_equal_triangle_order(self):
        original = ray_correspondences(self.vertices, self.triangles, [[50, 50]], self.camera)
        changed = np.array([[0, 3, 1], [1, 3, 2]])
        rebound = rebind_native_faces(self.triangles, changed, original)
        self.assertFalse(rebound["supported"][0])
        self.assertEqual(rebound["unmatched_landmark_indices"], [0])
