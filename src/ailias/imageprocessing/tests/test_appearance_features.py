import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageDraw

from ailias.imageprocessing.appearance_features import (
    IRIS_EYES, extract_bisenet_features, iris_colours, photo_features,
)
from ailias.imageprocessing.face_features import FaceFeatures, FaceFeaturesBiseNet, FaceRegionFeatures


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
    def test_all_natural_regions_retain_individual_colour_counts_and_bounds(self):
        """Measure each natural class independently from one label image, including formerly unused parts.
        Exclude accessories/body classes while preserving the previous combined brow and lip summaries.
        """
        labels = np.tile(np.repeat(np.arange(19, dtype=np.uint8), 8), (8, 1))
        classes = np.arange(19)
        colours = np.column_stack((20 + classes * 5, 80 + classes * 3, 140 + classes * 2)).astype(np.uint8)
        result = extract_bisenet_features(colours[labels], labels)
        self.assertIsInstance(result, FaceFeaturesBiseNet)
        expected = {
            "skin": 1, "left_eyebrow": 2, "right_eyebrow": 3, "left_eye": 4, "right_eye": 5,
            "left_ear": 7, "right_ear": 8, "nose": 10, "mouth_interior": 11,
            "upper_lip": 12, "lower_lip": 13, "scalp_hair": 17,
        }
        for name, label in expected.items():
            with self.subTest(region=name):
                region = getattr(result, name)
                self.assertIsInstance(region, FaceRegionFeatures)
                self.assertEqual(region.status, "observed")
                self.assertEqual(region.parser_classes, [label])
                self.assertEqual(region.region_pixels, 64)
                self.assertEqual(region.sample_pixels, 64)
                self.assertEqual(region.bounds_xy, [[label * 8, 0], [(label + 1) * 8, 8]])
                linear = ((colours[label].astype(float) / 255 + .055) / 1.055) ** 2.4
                np.testing.assert_allclose(region.median_linear_rgb, linear)
                self.assertTrue(region.photographed_lighting_retained)
                self.assertEqual(region.confidence, "uncalibrated_semantic_prediction")
        self.assertEqual(result.eyebrows.region_pixels, 128)
        self.assertEqual(result.lips.region_pixels, 128)
        self.assertEqual(set(result.to_dict()), set(expected) | {"eyebrows", "lips"})

    def test_missing_and_clipped_regions_retain_available_geometry_without_inventing_colour(self):
        """Preserve parser IDs, sample counts and bounds even when regional colour cannot be estimated.
        Distinguish an absent eye from a present nose with too few unclipped samples.
        """
        rgb = np.zeros((8, 8, 3), np.uint8)
        rgb[0, :4] = [120, 80, 60]
        result = extract_bisenet_features(rgb, np.full((8, 8), 10, np.uint8))
        self.assertEqual(result.nose.status, "review")
        self.assertEqual(result.nose.parser_classes, [10])
        self.assertEqual(result.nose.region_pixels, 64)
        self.assertEqual(result.nose.sample_pixels, 4)
        self.assertEqual(result.nose.bounds_xy, [[0, 0], [8, 8]])
        self.assertIsNone(result.nose.median_linear_rgb)
        self.assertEqual(result.nose.reason, "insufficient_unclipped_region_pixels")
        self.assertEqual(result.left_eye.region_pixels, 0)
        self.assertIsNone(result.left_eye.bounds_xy)
        self.assertIsNone(result.left_eye.median_linear_rgb)

    def test_structured_regions_and_complete_results_serialize_as_json_and_plain_dicts(self):
        """Serialize measured regions and their enclosing result without paths or nested class objects.
        Preserve the same data in JSON strings and dictionaries for ready, review and error records.
        """
        rgb, _ = eye_image()
        regions = extract_bisenet_features(rgb, np.ones(rgb.shape[:2], np.uint8))
        for value in (regions.skin, regions):
            with self.subTest(output=type(value).__name__):
                self.assertEqual(json.loads(str(value)), value.to_dict())
        for status in ("ready", "review", "error"):
            result = FaceFeatures(status=status, source=Path("fixture.png"), bisenet=regions,
                                  error_type="RuntimeError" if status == "error" else None)
            payload = result.to_dict()
            self.assertEqual(json.loads(str(result)), payload)
            self.assertEqual(payload["source"], "fixture.png")
            self.assertEqual(payload["bisenet"]["skin"], regions.skin.to_dict())
            self.assertEqual(payload["status"], status)
        self.assertEqual(json.loads(str(FaceFeatures()))["bisenet"], None)

    def test_bisenet_region_measurements_reject_invalid_pixels_and_label_grids(self):
        """Reject invalid RGB values, mismatched grids and unknown semantic IDs before sampling.
        Keep the standalone region extractor honest without requiring any neural runtime.
        """
        for rgb, labels in (
            (np.zeros((4, 4, 3), np.float32), np.zeros((4, 4), np.uint8)),
            (np.zeros((4, 4, 3), np.uint8), np.zeros((2, 2), np.uint8)),
            (np.zeros((4, 4, 3), np.uint8), np.full((4, 4), 19, np.uint8)),
        ):
            with self.subTest(rgb=rgb.shape, labels=labels.shape), self.assertRaises(ValueError):
                extract_bisenet_features(rgb, labels)

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
