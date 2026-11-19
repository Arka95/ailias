from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.imageprocessing.face_features import FaceExtractionResult, FaceFeatures, FaceQuality
from ailias.imageprocessing.face_grouping import (
    FaceGroupingPolicy, group_face_images, load_grouping_policy,
)


def extracted_orientation(yaw=0, pitch=0, roll=0):
    """Provide previously extracted orientation without computing a transform during grouping tests.
    Keep fixtures independent of inference and of the extraction stage's pose arithmetic.
    """
    return {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll, "calibrated": False}


class FaceGroupingTests(unittest.TestCase):
    def setUp(self):
        """Create synthetic source pixels and dense-coordinate evidence without neural runtimes.
        Isolate explicit review copies in a temporary directory owned by this test.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.review = self.root / "review"
        image = Image.new("RGB", (160, 128), (160, 110, 75))
        indices = np.arange(478)
        self.points = np.column_stack((32 + indices % 23 * 4, 13 + indices // 23 * 2)).astype(float)
        self.result = FaceExtractionResult(
            image=image, head_image=image.convert("RGBA"),
            features=FaceFeatures(landmarks={"source_xy": self.points.tolist()},
                                  facial_transform=np.eye(4).tolist(), orientation=extracted_orientation(),
                                  quality=FaceQuality()))
        self.policy = FaceGroupingPolicy()

    def group(self, inputs, **options):
        """Apply the public grouping API with this test's explicit review directory and policy.
        Permit narrowly scoped policy changes without loading models or user configuration.
        """
        return group_face_images(inputs, review_directory=self.review, policy=options.get("policy", self.policy))

    def test_all_usable_views_including_side_upper_lower_and_roll_are_kept(self):
        """Keep all usable members in view groups, including large and combined head angles.
        Apply no per-view quota and create no review directory for selected images.
        """
        views = (
            (0, 0, 0, "front"), (0, 0, 70, "front"),
            (35, 0, 0, "right_oblique"), (-35, 0, 0, "left_oblique"),
            (80, 0, 0, "right_profile"), (-80, 0, 0, "left_profile"),
            (0, 60, 0, "up"), (0, -60, 0, "down"), (-75, 35, 0, "left_profile_up"),
            (75, -35, 0, "right_profile_down"),
        )
        inputs = [replace(self.result, features=replace(
            self.result.features, orientation=extracted_orientation(yaw, pitch, roll)))
            for yaw, pitch, roll, _ in views]
        result = self.group(inputs)
        self.assertEqual(len(result.selected), len(inputs))
        self.assertEqual(result.excluded, [])
        self.assertEqual(set(result.groups), {entry[3] for entry in views})
        self.assertEqual(len(result.groups["front"]), 2)
        self.assertFalse(self.review.exists())
        self.assertEqual(json.loads(str(result)), result.to_dict())

    def test_missing_or_invalid_orientation_reports_unknown_without_excluding_images(self):
        """Consume unavailable orientation diagnostics without recalculating the raw facial transform.
        Keep otherwise valid facial extractions in the unknown view instead of discarding them.
        """
        orientations = [None, {"calibrated": False, "reason": "missing_or_invalid_pose"}]
        inputs = [replace(self.result, features=replace(self.result.features, orientation=value))
                  for value in orientations]
        result = self.group(inputs)
        self.assertEqual(len(result.groups["unknown"]), len(inputs))
        self.assertEqual(len(result.selected), len(inputs))
        self.assertEqual(result.excluded, [])
        for item in result.selected:
            self.assertEqual(item.orientation["reason"], "missing_or_invalid_pose")
            self.assertIsNone(item.exclusion_reason)
            self.assertIsNone(item.review_copy)
        self.assertFalse(self.review.exists())

    def test_grouping_reuses_extracted_quality_and_orientation_without_recalculation_or_rejection(self):
        """Keep already assessed evidence and warning-only inputs without running assessment again.
        Use stored angles even when the retained raw transform would describe another view.
        """
        quality = FaceQuality(laplacian_variance=0, supported_landmarks=0, max_expression_score=.9,
                              reasons=["face_blurred_or_low_detail", "insufficient_head_support",
                                       "non_neutral_expression"])
        features = replace(self.result.features, orientation=extracted_orientation(80, 35, 70), quality=quality)
        with patch("ailias.imageprocessing.features.face_quality", side_effect=AssertionError("recomputed")), patch(
            "ailias.imageprocessing.face_feature_extraction.determine_face_orientation",
            side_effect=AssertionError("recomputed"),
        ):
            result = self.group([replace(self.result, features=features)])
        self.assertIn("right_profile_up", result.groups)
        item = result.selected[0]
        self.assertIs(item.quality, quality)
        self.assertIs(item.orientation, features.orientation)
        self.assertEqual(item.to_dict()["quality"], quality.to_dict())
        self.assertFalse(self.review.exists())

    def test_group_before_selection_keeps_excluded_candidates_with_their_view(self):
        """Retain a failed candidate in its known view group alongside successful members.
        Copy only excluded images and keep explicit per-member decisions for inspection.
        """
        failed = replace(self.result, error_type="ValueError", reason="fixture failure")
        result = self.group([self.result, failed])
        self.assertEqual(len(result.groups["front"]), 2)
        self.assertEqual(len(result.selected), 1)
        self.assertEqual(len(result.excluded), 1)
        self.assertEqual(result.excluded[0].exclusion_reason, "fixture failure")
        self.assertTrue(result.excluded[0].review_copy.is_file())

    def test_outside_degenerate_and_invalid_coordinates_are_excluded_but_partial_clipping_is_kept(self):
        """Exclude invalid coordinate evidence or faces with no nonzero overlap against source pixels.
        Preserve a partially clipped face and annotate clipping rather than rejecting useful anatomy.
        """
        for points in (self.points + 400, np.zeros((478, 2)), [[1, 2]], np.full((478, 2), np.nan)):
            with self.subTest(shape=np.shape(points)):
                source = replace(self.result, features=replace(
                    self.result.features, landmarks={"source_xy": np.asarray(points).tolist()}))
                result = self.group([source])
                self.assertEqual(len(result.excluded), 1)
                self.assertIsNotNone(result.excluded[0].exclusion_reason)
        clipped = self.points.copy()
        clipped[0, 0] = -10
        source = replace(self.result, features=replace(
            self.result.features, landmarks={"source_xy": clipped.tolist()}))
        self.assertEqual(len(self.group([source]).selected), 1)

    def test_source_copies_preserve_original_bytes_and_never_overwrite_an_earlier_copy(self):
        """Copy an excluded source photo unchanged with a fresh destination for repeated inputs.
        Preserve original data and write neither feature JSON nor automatic masks into review.
        """
        path = self.root / "original.png"
        self.result.image.save(path)
        original = path.read_bytes()
        source = FaceExtractionResult(source=path, error_type="ValueError", reason="invalid fixture")
        copies = [item.review_copy for item in self.group([source, source]).excluded]
        self.assertEqual(len(set(copies)), 2)
        self.assertEqual(path.read_bytes(), original)
        for copy in copies:
            self.assertNotEqual(copy, path)
            self.assertEqual(copy.read_bytes(), original)
        self.assertEqual(set(self.review.iterdir()), set(copies))

    def test_video_frames_and_in_memory_inputs_export_png_instead_of_copying_video_bytes(self):
        """Export excluded frame pixels with their frame metadata instead of copying the source video.
        Handle caller-owned in-memory images identically without changing any source pixels.
        """
        video = self.root / "clip.mp4"
        video.write_bytes(b"fixture video, not decoded")
        frame = FaceExtractionResult(source=video, image=self.result.image, frame_index=12,
                                     timestamp_seconds=1.5, error_type="RuntimeError", reason="fixture")
        memory = replace(frame, source=None, frame_index=None, timestamp_seconds=None)
        result = self.group([frame, memory])
        for item in result.excluded:
            self.assertEqual(item.review_copy.suffix, ".png")
            with Image.open(item.review_copy) as image:
                self.assertEqual(image.tobytes(), self.result.image.tobytes())
        self.assertEqual(result.excluded[0].extraction.frame_index, 12)
        self.assertEqual(result.excluded[0].extraction.timestamp_seconds, 1.5)
        self.assertEqual(video.read_bytes(), b"fixture video, not decoded")

    def test_copy_failure_is_reported_per_item_and_does_not_stop_later_members(self):
        """Report missing pixels/files or an unusable review directory without dropping failed records.
        Continue selecting later valid inputs and preserve the blocking file unchanged.
        """
        missing = FaceExtractionResult(source=self.root / "missing.png", error_type="ValueError", reason="fixture")
        result = self.group([missing, self.result])
        self.assertIn("FileNotFoundError", result.excluded[0].review_copy_error)
        self.assertEqual(len(result.selected), 1)
        self.review.write_bytes(b"block directory creation")
        failed = replace(self.result, error_type="ValueError", reason="fixture")
        blocked = self.group([failed, self.result])
        self.assertIsNotNone(blocked.excluded[0].review_copy_error)
        self.assertIsNone(blocked.excluded[0].review_copy)
        self.assertEqual(len(blocked.selected), 1)
        self.assertEqual(self.review.read_bytes(), b"block directory creation")

    def test_partial_review_export_is_cleaned_up_after_encoding_failure(self):
        """Remove a partial review PNG when its encoding fails without touching the original pixels.
        Keep the failed candidate and copy-error report available to the caller.
        """
        def fail_save(output, **options):
            """Simulate an encoder writing some bytes before failing.
            Exercise cleanup without a real filesystem-permission or codec failure.
            """
            output.write(b"partial")
            raise OSError("fixture encoder failure")

        with patch.object(self.result.image, "save", side_effect=fail_save):
            result = self.group([replace(self.result, error_type="ValueError", reason="fixture")])
        self.assertIn("OSError", result.excluded[0].review_copy_error)
        self.assertEqual(list(self.review.iterdir()), [])

    def test_grouping_configuration_changes_view_bins_without_discarding_images(self):
        """Load named grouping settings and apply explicit view-bin overrides independently of extraction.
        Treat changed boundaries as grouping choices rather than face validity or pose limits.
        """
        path = self.root / "grouping.json"
        path.write_text('{"frontal_yaw_degrees": 30}', encoding="utf-8")
        source = replace(self.result, features=replace(self.result.features, orientation=extracted_orientation(25)))
        self.assertIn("right_oblique", self.group([source]).groups)
        self.assertIn("front", self.group([source], policy=load_grouping_policy(path)).groups)
        self.assertEqual(len(group_face_images([source], review_directory=self.review).selected), 1)
        for changes in ({"frontal_yaw_degrees": 56}, {"profile_yaw_degrees": 91},
                        {"vertical_pitch_degrees": 0}, {"vertical_pitch_degrees": 91},
                        {"frontal_yaw_degrees": True}, {"vertical_pitch_degrees": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.policy, **changes)
