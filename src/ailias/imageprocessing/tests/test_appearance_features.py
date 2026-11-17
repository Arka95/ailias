import unittest

import numpy as np
from PIL import Image, ImageDraw

from ailias.imageprocessing.appearance_features import IRIS_EYES, iris_colours, photo_features


def eye_image():
    image = Image.new("RGB", (128, 64), (220, 160, 120))
    draw = ImageDraw.Draw(image)
    points = np.full((478, 2), [64., 32.])
    for center_x, (iris_index, eyelid) in zip((32, 96), IRIS_EYES):
        center = np.array([center_x, 32])
        points[iris_index:iris_index+5] = center + np.array(
            [[0, 0], [12, 0], [0, -12], [-12, 0], [0, 12]])
        angles = np.linspace(0, 2*np.pi, len(eyelid), endpoint=False)
        points[list(eyelid)] = center + np.c_[18*np.cos(angles), 13*np.sin(angles)]
        draw.ellipse((center_x-12, 20, center_x+12, 44), fill=(60, 35, 18))
        draw.ellipse((center_x-5, 27, center_x+5, 37), fill=(0, 0, 0))
        draw.rectangle((center_x+6, 28, center_x+8, 30), fill="white")
    points[33], points[263] = [14, 32], [114, 32]
    return np.asarray(image), points


class AppearanceFeatureTests(unittest.TestCase):
    def test_pupil_skin_and_highlights_do_not_replace_iris_colour(self):
        rgb, points = eye_image()
        result = iris_colours(rgb, points)
        expected = ((np.array([60, 35, 18])/255+.055)/1.055)**2.4
        np.testing.assert_allclose(result["median_linear_rgb"], expected)
        self.assertEqual(len(result["eyes"]), 2)

    def test_occluded_or_degenerate_iris_does_not_invent_colour(self):
        rgb, points = eye_image()
        for image, landmarks in ((np.zeros_like(rgb), points), (rgb, np.zeros_like(points))):
            with self.subTest(), self.assertRaises(ValueError):
                iris_colours(image, landmarks)

    def test_region_colour_excludes_clipping_and_preserves_feature_specific_abstention(self):
        rgb, points = eye_image()
        rgb = rgb.copy()
        rgb[:10] = [120, 80, 60]
        rgb[0, :10] = 255
        labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
        labels[:10] = 1
        result = photo_features(rgb, points, labels, np.ones(478, dtype=bool))
        skin = result["regions"]["skin"]
        expected = ((np.array([120, 80, 60])/255+.055)/1.055)**2.4
        np.testing.assert_allclose(skin["median_linear_rgb"], expected)
        self.assertEqual(skin["sample_pixels"], 1270)
        self.assertEqual(result["regions"]["scalp_hair"]["status"], "review")
        self.assertEqual(result["unresolved"]["beard"], "no_distinct_class_in_current_parser")
        self.assertFalse(result["intrinsic_colour_calibrated"])

    def test_photo_geometry_is_translation_invariant_and_occlusion_is_not_silently_ignored(self):
        rgb, points = eye_image()
        labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
        visible = np.ones(478, dtype=bool)
        first = photo_features(rgb, points, labels, visible)
        padded = np.pad(rgb, ((10, 0), (10, 0), (0, 0)))
        second = photo_features(padded, points+10, np.pad(labels, ((10, 0), (10, 0))), visible)
        self.assertEqual(first["geometry_2d"], second["geometry_2d"])
        visible[33] = False
        result = photo_features(rgb, points, labels, visible)
        self.assertEqual(result["geometry_2d"]["outer_eye_span"]["status"], "review")
        self.assertEqual(result["iris"]["status"], "review")
        self.assertFalse(result["metric_depth_measured"])

    def test_mismatched_labels_or_nonfinite_landmarks_fail_closed(self):
        rgb, points = eye_image()
        for labels in (np.zeros((2, 2), dtype=np.uint8), np.full(rgb.shape[:2], 19, dtype=np.uint8)):
            with self.subTest(), self.assertRaises(ValueError):
                photo_features(rgb, points, labels, np.ones(478, dtype=bool))
        points[0] = np.nan
        with self.assertRaises(ValueError):
            photo_features(rgb, points, np.zeros(rgb.shape[:2], dtype=np.uint8),
                           np.ones(478, dtype=bool))
