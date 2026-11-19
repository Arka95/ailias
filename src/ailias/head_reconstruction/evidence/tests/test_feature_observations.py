"""Verify retained facial evidence conversion without pixels or model inference."""

from dataclasses import replace
import unittest

import numpy as np

from ailias.head_reconstruction.evidence.feature_observations import observations_from_face_features
from ailias.imageprocessing.face_features import FaceFeatures, FaceQuality


class FeatureObservationTests(unittest.TestCase):
    def setUp(self):
        """Build one non-square coordinate grid with finite points and alpha support.
        Leave image pixels, model assets and orientation entirely unavailable.
        """
        indices = np.arange(478)
        points = np.column_stack((20 + indices % 23 * 4, 10 + indices // 23 * 2))
        self.features = FaceFeatures(
            source_size=(160, 128),
            landmarks={"source_xy": points.tolist(), "on_head": [True] * 478},
            quality=FaceQuality(reasons=["face_blurred_or_low_detail", "non_neutral_expression"]),
        )

    def test_conversion_keeps_native_grid_and_view_ids_without_requiring_pose_or_quality_approval(self):
        """Retain full pixel coordinates and derive outer-eye spacing from existing predictions.
        Do not reinterpret quality warnings as exclusions or synthesize missing camera calibration.
        """
        observation = observations_from_face_features({"photo:rear-camera": self.features})[0]
        self.assertEqual(observation["status"], "ready")
        self.assertEqual(observation["face_id"], "photo:rear-camera")
        self.assertEqual(observation["source_size"], [160, 128])
        self.assertEqual(observation["source_xy"], self.features.landmarks["source_xy"])
        points = np.asarray(self.features.landmarks["source_xy"])
        self.assertAlmostEqual(observation["eye_span_source"], np.linalg.norm(points[33] - points[263]))
        self.assertFalse(observation["landmark_depth_is_measured"])
        self.assertFalse(observation["per_point_visibility_available"])

    def test_outside_landmarks_lose_support_without_modifying_stored_predictions(self):
        """Prevent out-of-image points from acquiring support through a malformed stored mask.
        Preserve partially clipped views and leave the caller's coordinate/support arrays unchanged.
        """
        self.features.landmarks["source_xy"][0] = [-10, 5]
        observation = observations_from_face_features({"clipped": self.features})[0]
        self.assertEqual(observation["status"], "ready")
        self.assertFalse(observation["on_head"][0])
        self.assertTrue(self.features.landmarks["on_head"][0])
        self.assertEqual(self.features.landmarks["source_xy"][0], [-10, 5])

    def test_invalid_records_are_accounted_for_without_stopping_later_views(self):
        """Preserve each invalid view and its specific reason rather than guessing missing evidence.
        Continue conversion for later valid views without requiring their source images.
        """
        cases = [
            (None, "missing_face_features"),
            (replace(self.features, source_size=None), "missing_or_invalid_source_size"),
            (replace(self.features, source_size=(160, 0)), "missing_or_invalid_source_size"),
            (replace(self.features, source_size=(160.5, 128)), "missing_or_invalid_source_size"),
            (replace(self.features, source_size=(True, False)), "missing_or_invalid_source_size"),
            (replace(self.features, source_size=(160, True)), "missing_or_invalid_source_size"),
            (replace(self.features, landmarks=None), "missing_or_invalid_landmark_geometry"),
            (replace(self.features, landmarks={"source_xy": [[0, 0]], "on_head": [True]}),
             "invalid_dense_landmarks"),
        ]
        for field, value, reason in (
            ("source_xy", [[float("nan"), 0]] * 478, "invalid_dense_landmarks"),
            ("source_xy", [[0, 0]] * 478, "face_outside_image_or_degenerate_landmarks"),
            ("source_xy", [[900, 900]] * 478, "face_outside_image_or_degenerate_landmarks"),
            ("on_head", [True], "invalid_landmark_support"),
            ("on_head", [1] * 478, "invalid_landmark_support"),
        ):
            cases.append((replace(self.features, landmarks={**self.features.landmarks, field: value}), reason))
        coincident = [point[:] for point in self.features.landmarks["source_xy"]]
        coincident[263] = coincident[33][:]
        cases.append((replace(self.features, landmarks={**self.features.landmarks, "source_xy": coincident}),
                      "degenerate_outer_eye_span"))
        for features, reason in cases:
            with self.subTest(reason=reason):
                bad, good = observations_from_face_features({"bad": features, "good": self.features})
                self.assertEqual(bad["status"], "review")
                self.assertEqual(bad["reason"], reason)
                self.assertEqual(good["status"], "ready")
