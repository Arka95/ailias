from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.imageprocessing.face_features import FaceExtractionResult, FaceFeatures
from ailias.imageprocessing.face_grouping import (
    FaceGroupingPolicy, determine_face_orientation, group_face_images, load_grouping_policy,
)


def face_transform(yaw=0, pitch=0, roll=0, scale=1):
    """Build a synthetic scaled Rz/Ry/Rx transform with independent Euler angles.
    Exercise transform arithmetic without claiming a real photo's physical head orientation.
    """
    yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = scale * rz @ ry @ rx, [1, 2, 3]
    return result.tolist()


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
                                  facial_transform=face_transform()))
        self.policy = FaceGroupingPolicy()

    def group(self, inputs, **options):
        """Apply the public grouping API with this test's explicit review directory and policy.
        Permit narrowly scoped policy changes without loading models or user configuration.
        """
        return group_face_images(inputs, review_directory=self.review, policy=options.get("policy", self.policy))

    def test_scaled_transform_preserves_orientation_and_translation_does_not_change_angles(self):
        """Remove MediaPipe-compatible positive uniform scale before interpreting face orientation.
        Recover independent yaw/pitch/roll angles rather than rejecting a valid scaled rotation.
        """
        for scale in (0.25, 1, 3.5):
            with self.subTest(scale=scale):
                pose = determine_face_orientation(face_transform(-75, 35, 40, scale), self.policy)
                self.assertEqual(pose["category"], "left_profile_up")
                for name, angle in (("yaw", -75), ("pitch", 35), ("roll", 40)):
                    self.assertAlmostEqual(pose[name + "_deg"], angle)
                self.assertFalse(pose["calibrated"])

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
            self.result.features, facial_transform=face_transform(yaw, pitch, roll)))
            for yaw, pitch, roll, _ in views]
        result = self.group(inputs)
        self.assertEqual(len(result.selected), len(inputs))
        self.assertEqual(result.excluded, [])
        self.assertEqual(set(result.groups), {entry[3] for entry in views})
        self.assertEqual(len(result.groups["front"]), 2)
        self.assertFalse(self.review.exists())
        self.assertEqual(json.loads(str(result)), result.to_dict())

    def test_missing_or_invalid_orientation_reports_unknown_without_excluding_images(self):
        """Report malformed, missing, reflected, sheared or nonuniform transforms without raising errors.
        Keep otherwise valid facial extractions in the unknown view instead of discarding them.
        """
        shear = np.eye(4)
        shear[0, 1] = 0.3
        perspective = np.eye(4)
        perspective[3, 0] = 1
        transforms = [None, "invalid", [[1]], np.zeros((4, 4)).tolist(),
                      np.diag([1, 2, 1, 1]).tolist(), np.diag([-1, 1, 1, 1]).tolist(),
                      np.full((4, 4), np.nan).tolist(), shear.tolist(), perspective.tolist()]
        inputs = [replace(self.result, features=replace(self.result.features, facial_transform=value))
                  for value in transforms]
        result = self.group(inputs)
        self.assertEqual(len(result.groups["unknown"]), len(inputs))
        self.assertEqual(len(result.selected), len(inputs))
        self.assertEqual(result.excluded, [])
        for item in result.selected:
            self.assertEqual(item.orientation["reason"], "missing_or_invalid_pose")
            self.assertIsNone(item.exclusion_reason)
            self.assertIsNone(item.review_copy)
        self.assertFalse(self.review.exists())

    def test_quality_warnings_never_exclude_low_detail_exposure_or_expression(self):
        """Annotate unusably lit-looking or expressive inputs without overriding the keep-usable policy.
        Keep quality thresholds separate from extraction validity and retain every view member.
        """
        for colour, warning in (("black", "face_underexposed"), ("white", "face_overexposed"),
                                ("gray", "face_blurred_or_low_detail")):
            with self.subTest(colour=colour):
                source = replace(self.result, image=Image.new("RGB", (160, 128), colour),
                                 features=replace(self.result.features, blendshapes={"jawOpen": .9}))
                item = self.group([source]).selected[0]
                self.assertIn(warning, item.quality["reasons"])
                self.assertIn("non_neutral_expression", item.quality["reasons"])
                permissive = replace(self.policy, max_neutral_expression=.95)
                adjusted = self.group([source], policy=permissive).selected[0]
                self.assertNotIn("non_neutral_expression", adjusted.quality["reasons"])
        self.assertFalse(self.review.exists())

    def test_alpha_and_supported_count_thresholds_change_annotations_not_features_or_selection(self):
        """Recompute quality support from prepared alpha with grouping-owned opacity/count thresholds.
        Preserve raw facial evidence and cutout pixels while changing only the warning decision.
        """
        head = self.result.head_image.copy()
        head.putalpha(230)
        source = replace(self.result, head_image=head)
        original_pixels = head.tobytes()
        item = self.group([source]).selected[0]
        self.assertIn("insufficient_head_support", item.quality["reasons"])
        permissive = replace(self.policy, min_landmark_alpha=220)
        adjusted = self.group([source], policy=permissive).selected[0]
        self.assertNotIn("insufficient_head_support", adjusted.quality["reasons"])
        self.assertIs(adjusted.extraction.features, source.features)
        self.assertEqual(head.tobytes(), original_pixels)
        alpha = Image.new("L", head.size)
        for x, y in self.points[:100].astype(int):
            alpha.putpixel((int(x), int(y)), 255)
        head.putalpha(alpha)
        self.assertIn("insufficient_head_support", self.group([source]).selected[0].quality["reasons"])
        fewer_required = replace(self.policy, min_supported_landmarks=50)
        self.assertNotIn("insufficient_head_support",
                         self.group([source], policy=fewer_required).selected[0].quality["reasons"])

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
        self.assertIn("face_clipped", self.group([source]).selected[0].quality["reasons"])

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
        source = replace(self.result, features=replace(self.result.features, facial_transform=face_transform(25)))
        self.assertIn("right_oblique", self.group([source]).groups)
        self.assertIn("front", self.group([source], policy=load_grouping_policy(path)).groups)
        self.assertEqual(len(group_face_images([source], review_directory=self.review).selected), 1)
        for changes in ({"min_landmark_alpha": 0}, {"min_landmark_alpha": 256}, {"min_landmark_alpha": 1.5},
                        {"min_supported_landmarks": 0}, {"min_supported_landmarks": 469},
                        {"max_neutral_expression": 1.1}, {"max_neutral_expression": float("nan")},
                        {"frontal_yaw_degrees": 56}, {"profile_yaw_degrees": 91},
                        {"vertical_pitch_degrees": 0}, {"vertical_pitch_degrees": 91},
                        {"max_dark_fraction": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.policy, **changes)
