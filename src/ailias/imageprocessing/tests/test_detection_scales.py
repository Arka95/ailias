import unittest

import cv2
import numpy as np

from ailias.imageprocessing.backend import detect_scaled_rows


class DetectionScaleTests(unittest.TestCase):
    def test_rescues_a_large_face_and_restores_native_boxes_and_landmarks(self):
        sizes = []
        row = np.array([20, 30, 40, 50, 25, 40, 45, 40, 35, 50, 28, 65, 42, 65, .95],
                       dtype=np.float32)
        def infer(image):
            sizes.append(image.shape[:2])
            return [] if len(sizes) == 1 else [row]
        result = detect_scaled_rows(cv2, np.zeros((2027, 1920, 3), np.uint8), 1600, infer)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(sizes), 2)
        np.testing.assert_allclose(result[0][:14:2], row[:14:2]*1920/sizes[1][1])
        np.testing.assert_allclose(result[0][1:14:2], row[1:14:2]*2027/sizes[1][0])
        self.assertAlmostEqual(result[0][14], .95)

    def test_successful_first_scale_is_unchanged_and_not_reclassified(self):
        calls = []
        row = np.ones(15, np.float32)
        def infer(image):
            calls.append(image.shape)
            return [row]
        result = detect_scaled_rows(cv2, np.zeros((100, 120, 3), np.uint8), 1600, infer)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(result[0], row)

    def test_small_empty_images_are_not_repeated_or_upscaled(self):
        calls = []
        def infer(image):
            calls.append(image.shape)
            return None
        self.assertEqual(detect_scaled_rows(cv2, np.zeros((100, 120, 3), np.uint8), 1600, infer), [])
        self.assertEqual(calls, [(100, 120, 3)])

    def test_inference_failure_propagates_without_trying_another_scale_or_device(self):
        calls = []
        def infer(image):
            calls.append(image.shape)
            raise RuntimeError("fixture inference failure")
        with self.assertRaisesRegex(RuntimeError, "fixture inference"):
            detect_scaled_rows(cv2, np.zeros((2000, 1200, 3), np.uint8), 1600, infer)
        self.assertEqual(len(calls), 1)
