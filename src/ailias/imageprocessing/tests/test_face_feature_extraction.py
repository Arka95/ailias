from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ailias.imageprocessing.face_feature_extraction import (
    FaceExtractionResult,
    FaceFeatureExtractor,
    FaceFeatures,
    FaceFeaturesBiseNet,
    FaceQuality,
)
from ailias.imageprocessing.settings import FaceQualityPolicy, load_face_quality_policy


def face_transform(yaw=0, pitch=0, roll=0, scale=1):
    """Build a synthetic scaled Rz/Ry/Rx transform with independent Euler angles.
    Exercise transform arithmetic without claiming a real photo's physical head orientation.
    """
    yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    rz = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]]
    )
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = scale * rz @ ry @ rx, [1, 2, 3]
    return result.tolist()


def dense_prediction():
    """Provide finite normalized landmarks and neutral expression/transform evidence.
    Distribute points over synthetic skin without claiming an anatomical fixture.
    """
    indices = np.arange(478)
    points = np.column_stack(
        (0.2 + indices % 23 / 22 * 0.6, 0.1 + indices // 23 / 21 * 0.3, np.zeros(478))
    )
    return {
        "faces": [points.tolist()],
        "blendshapes": [{"mouthSmileLeft": 0.1}],
        "transforms": [np.eye(4).tolist()],
    }


class FaceFeatureExtractorTests(unittest.TestCase):
    def setUp(self):
        """Create prepared native-grid pixels and a fake MediaPipe task without preparation models.
        Patch inference so units require neither real weights nor GPU execution.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model = self.root / "mediapipe.model"
        model.write_bytes(b"unit fixture, not model weights")
        self.landmarker = Mock()
        self.landmarker.detect.side_effect = lambda rgb: dense_prediction()
        self.create_landmarker = self.enterContext(
            patch(
                "ailias.imageprocessing.dense.DenseDetector",
                return_value=self.landmarker,
            )
        )
        self.extractor = FaceFeatureExtractor(mediapipe_model=model, device="cpu")
        self.addCleanup(self.extractor.close)
        y, x = np.indices((128, 160))
        values = ((x * 41 + y * 31) % 140 + 50).astype(np.uint8)
        self.image = Image.fromarray(values).convert("RGB")

    def prepared_image(self, image):
        """Provide independent preparation products on the supplied image's pixel grid.
        Stand in for a caller without constructing MODNet, BiSeNet or an ImageProcessor.
        """
        labels = np.full((image.height, image.width), 16, np.uint8)
        labels[: image.height // 2] = 1
        labels[: image.height // 16] = 17
        head = image.convert("RGBA")
        head.putalpha(Image.fromarray(np.where(labels == 16, 0, 255).astype(np.uint8)))
        return {
            "image": head,
            "original_image": image.copy(),
            "regions": Image.fromarray(labels),
            "source": self.root / "video.mp4",
            "frame_index": 12,
            "timestamp_seconds": 1.5,
        }

    def extract_image(self, image):
        """Pass existing pixels and region labels into the feature API for review tests.
        Keep the fixture independent of preparation model availability and lifecycle.
        """
        return self.extractor.extract_features(self.prepared_image(image))

    def test_features_use_existing_labels_native_coordinates_and_original_colours(self):
        """Extract original-colour evidence from a visibly corrected head without rerunning CNN preparation.
        Preserve source identity, input pixels and labels without writing sidecars.
        """
        prepared = self.prepared_image(self.image)
        prepared["image"].paste((0, 0, 255), (0, 0, 160, 64))
        before = {
            key: prepared[key].tobytes()
            for key in ("image", "original_image", "regions")
        }
        files = set(self.root.iterdir())
        with patch(
            "cv2.dnn.readNetFromONNX",
            side_effect=AssertionError("Preparation was rerun"),
        ):
            result = self.extractor.extract_features(prepared)
        self.assertIsInstance(result, FaceFeatures)
        normalized = np.asarray(result.landmarks["normalized_predicted"])
        np.testing.assert_allclose(
            result.landmarks["source_xy"], normalized[:, :2] * self.image.size
        )
        skin = result.features["regions"]["skin"]
        self.assertIsInstance(result.bisenet, FaceFeaturesBiseNet)
        self.assertEqual(result.bisenet.skin.to_dict(), skin)
        self.assertEqual(skin["status"], "observed")
        self.assertAlmostEqual(
            skin["median_linear_rgb"][0], skin["median_linear_rgb"][2]
        )
        self.assertEqual(result.features_colour_source, "original_rgb")
        self.assertFalse(result.features["metric_depth_measured"])
        hair = result.features["facial_hair_appearance"]
        self.assertFalse(hair["distinct_parser_class"])
        self.assertFalse(hair["intrinsic_reflectance_measured"])
        self.assertEqual(set(hair["regions"]), {"beard", "moustache"})
        for key, pixels in before.items():
            self.assertEqual(prepared[key].tobytes(), pixels)
        self.assertEqual(set(self.root.iterdir()), files)

    def test_direct_result_retains_pixels_and_propagates_runtime_failure(self):
        prepared = self.prepared_image(self.image)
        result = self.extractor.extract_feature_result(prepared)
        self.assertIs(result.image, prepared["original_image"])
        self.assertEqual(result.features.source_size, prepared["image"].size)
        self.landmarker.detect.side_effect = RuntimeError("model failure")
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            self.extractor.extract_feature_result(prepared)

    def test_feature_roundtrip_retains_geometry_grid_and_typed_assessments_without_pixels(
        self,
    ):
        """Keep non-square pixel dimensions with the landmarks after strict-JSON serialization.
        Restore retained predictions, regions and quality without loading pixels or rerunning extraction.
        """
        features = self.extract_image(self.image)
        with patch(
            "PIL.Image.open", side_effect=AssertionError("Pixels were reloaded")
        ):
            restored = FaceFeatures.from_dict(json.loads(str(features)))
        self.assertEqual(restored, features)
        self.assertEqual(restored.source_size, (160, 128))
        self.assertEqual(restored.to_dict()["source_size"], [160, 128])
        self.assertIsInstance(restored.bisenet, FaceFeaturesBiseNet)
        self.assertIsInstance(restored.quality, FaceQuality)
        self.assertEqual(self.landmarker.detect.call_count, 1)

    def test_landmarker_is_lazy_reused_and_closed_after_feature_collection(self):
        """Reuse only the owned landmark task across streamed feature extraction.
        Closing preparation is not part of this object's lifecycle.
        """
        prepared = self.prepared_image(self.image)
        results = self.extractor.extract_features_batch([prepared, prepared])
        self.create_landmarker.assert_not_called()
        first, second = next(results), next(results)
        self.assertIsNone(first.error_type)
        self.assertIsNotNone(second.features.landmarks)
        self.assertEqual(first.source, prepared["source"])
        self.assertEqual(first.frame_index, 12)
        self.assertEqual(first.timestamp_seconds, 1.5)
        self.assertIs(first.image, prepared["original_image"])
        self.create_landmarker.assert_called_once_with(
            self.root / "mediapipe.model", device="cpu"
        )
        self.extractor.close()
        self.extractor.close()
        self.landmarker.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.extractor.extract_features(prepared)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with self.extractor:
                pass

    def test_transparent_rgb_is_hidden_from_mediapipe(self):
        """Hide transparent source colours before MediaPipe's face and landmark inference.
        Feed a neutral-grey composite rather than arbitrary RGB under zero opacity.
        """
        self.extractor.detect_face_landmarks(
            Image.new("RGBA", (128, 128), (250, 0, 0, 0))
        )
        rgb = self.landmarker.detect.call_args.args[0]
        np.testing.assert_array_equal(rgb, np.full((128, 128, 3), 127, np.uint8))

    def test_empty_head_skips_landmarker_loading(self):
        """Return an extraction failure when prepared alpha contains no head foreground.
        Avoid constructing the neural task for an empty cutout.
        """
        prepared = self.prepared_image(self.image)
        prepared["image"].putalpha(0)
        result = next(self.extractor.extract_features_batch([prepared]))
        self.assertIsInstance(result, FaceExtractionResult)
        self.assertEqual(result.reason, "no_head_foreground")
        self.assertFalse(result.execution["mediapipe_executed"])
        self.create_landmarker.assert_not_called()

    def test_mismatched_or_missing_preparation_products_fail_before_inference(self):
        """Reject missing pixels, missing labels and inconsistent preparation coordinate grids.
        Do not spend inference work on evidence that cannot be sampled honestly.
        """
        for field, value in (
            ("image", self.image),
            ("original_image", None),
            ("regions", self.image),
            ("original_image", Image.new("RGB", (10, 10))),
            ("regions", Image.new("L", (10, 10))),
        ):
            with self.subTest(field=field):
                prepared = {**self.prepared_image(self.image), field: value}
                with self.assertRaises(ValueError):
                    self.extractor.extract_features(prepared)
        self.create_landmarker.assert_not_called()

    def test_batch_forwards_preparation_errors_and_continues_after_extraction_failure(
        self,
    ):
        """Preserve preparation failures and isolate failed landmark inference to its input.
        Reuse the task for the next prepared item without rerunning any preparation stage.
        """
        prepared = self.prepared_image(self.image)
        error = {
            "status": "error",
            "error_type": "FileNotFoundError",
            "reason": "fixture missing input",
            "source": "missing.png",
        }
        self.landmarker.detect.side_effect = [
            RuntimeError("fixture"),
            dense_prediction(),
        ]
        results = list(
            self.extractor.extract_features_batch([error, prepared, prepared])
        )
        for result in results:
            self.assertIsInstance(result, FaceExtractionResult)
        self.assertEqual(results[0].source, error["source"])
        self.assertEqual(results[0].error_type, error["error_type"])
        self.assertEqual(results[0].reason, error["reason"])
        self.assertIsNone(results[0].features)
        self.assertEqual(results[1].error_type, "RuntimeError")
        self.assertEqual(results[1].reason, "fixture")
        self.assertEqual(results[1].frame_index, 12)
        self.assertIsNone(results[2].error_type)
        self.assertIsNotNone(results[2].features.landmarks)
        self.create_landmarker.assert_called_once()

    def test_feature_visibility_records_positive_alpha_without_collection_quality_thresholds(
        self,
    ):
        """Retain faint visible pixels as raw support rather than applying a quality-selection threshold.
        Retain stricter opacity warnings separately in FaceQuality without changing head pixels.
        """
        prepared = self.prepared_image(self.image)
        alpha = np.asarray(prepared["image"].getchannel("A")).copy()
        alpha[alpha != 0] = 230
        prepared["image"].putalpha(Image.fromarray(alpha))
        before = prepared["image"].tobytes()
        result = self.extractor.extract_features(prepared)
        self.assertTrue(all(result.landmarks["on_head"]))
        self.assertEqual(result.quality.supported_landmarks, 0)
        self.assertIn("insufficient_head_support", result.quality.reasons)
        self.extractor.quality_policy = replace(
            self.extractor.quality_policy, min_landmark_alpha=220
        )
        adjusted = self.extractor.extract_features(prepared)
        self.assertEqual(adjusted.quality.supported_landmarks, 468)
        self.assertNotIn("insufficient_head_support", adjusted.quality.reasons)
        self.assertEqual(prepared["image"].tobytes(), before)

    def test_default_cuda_keeps_explicit_windows_cpu_exception(self):
        """Select MediaPipe's CPU delegate on Windows even in the default CUDA mode.
        Verify the documented exception without probing hardware or silently retrying.
        """
        with patch(
            "ailias.imageprocessing.face_feature_extraction.platform.system",
            return_value="Windows",
        ):
            extractor = FaceFeatureExtractor(
                mediapipe_model=self.root / "mediapipe.model"
            )
        with extractor:
            result = next(
                extractor.extract_features_batch([self.prepared_image(self.image)])
            )
        self.create_landmarker.assert_called_once_with(
            self.root / "mediapipe.model", device="cpu"
        )
        self.assertEqual(
            result.execution,
            {
                "requested_device": "cuda",
                "mediapipe_device": "cpu",
                "mediapipe_executed": True,
            },
        )

    def test_missing_model_and_exception_cleanup_do_not_load_or_download_weights(self):
        """Fail on missing local weights and release an owned task after context exceptions.
        Constructing an extractor never starts inference, downloads models or launches processes.
        """
        with FaceFeatureExtractor(device="cpu") as extractor:
            with self.assertRaisesRegex(ValueError, "mediapipe"):
                extractor.extract_features(self.prepared_image(self.image))
        with FaceFeatureExtractor(
            mediapipe_model=self.root / "missing.task", device="cpu"
        ) as extractor:
            with self.assertRaises(FileNotFoundError):
                extractor.detect_face_landmarks(self.image)
        self.create_landmarker.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with self.extractor:
                self.extractor.detect_face_landmarks(self.image)
                raise RuntimeError("fixture")
        self.landmarker.close.assert_called_once()
        with self.assertRaisesRegex(ValueError, "device"):
            FaceFeatureExtractor(device="auto")

    def test_no_face_multiple_faces_and_invalid_points_abstain_without_features(self):
        """Require one valid dense face rather than silently selecting a subject.
        Keep the prepared cutout available while returning explicit review reasons.
        """
        for faces, reason in (
            ([], "no_face_detected"),
            (dense_prediction()["faces"] * 2, "ambiguous_face_detections"),
            ([[[float("nan"), 0, 0]] * 478], "invalid_dense_landmarks"),
        ):
            with self.subTest(reason=reason):
                self.landmarker.detect.side_effect = None
                self.landmarker.detect.return_value = {
                    "faces": faces,
                    "blendshapes": [],
                    "transforms": [],
                }
                result = next(
                    self.extractor.extract_features_batch(
                        [self.prepared_image(self.image)]
                    )
                )
                self.assertEqual(result.reason, reason)
                self.assertIsNone(result.features.features)
                self.assertIsInstance(result.features.bisenet, FaceFeaturesBiseNet)
                self.assertEqual(result.features.bisenet.skin.status, "observed")
                self.assertIsNone(result.features.quality)

    def test_outside_and_clipped_landmarks_preserve_distinct_review_outcomes(self):
        """Retain structurally valid point predictions even when outside or clipped against the image.
        Annotate their quality without taking over grouping's keep/exclude decision.
        """
        for outside in (True, False):
            with self.subTest(outside=outside):
                detection = dense_prediction()
                if outside:
                    for point in detection["faces"][0]:
                        point[0] += 2
                else:
                    detection["faces"][0][0][0] = -0.1
                self.landmarker.detect.side_effect = None
                self.landmarker.detect.return_value = detection
                result = self.extract_image(self.image)
                self.assertIsNotNone(result.features)
                self.assertEqual(
                    result.landmarks["normalized_predicted"], detection["faces"][0]
                )
                self.assertIn("face_clipped", result.quality.reasons)
                self.assertEqual(
                    result.quality.clipped_landmarks, 468 if outside else 1
                )
                self.assertEqual(result.quality.native_short_side is None, outside)

    def test_extraction_retains_typed_quality_metrics_and_expression_evidence_as_strict_json(
        self,
    ):
        """Compute structured quality before grouping and preserve scores/counts, not only warning strings.
        Measure original pixels even when the prepared head's visible colour is different.
        """
        detection = dense_prediction()
        detection["blendshapes"][0]["jawOpen"] = 0.9
        self.landmarker.detect.side_effect = None
        self.landmarker.detect.return_value = detection
        for colour, fraction, warning in (
            ("black", "dark_fraction", "face_underexposed"),
            ("white", "bright_fraction", "face_overexposed"),
        ):
            with self.subTest(colour=colour):
                prepared = self.prepared_image(
                    Image.new("RGB", self.image.size, colour)
                )
                prepared["image"].paste((120, 120, 120, 255), (0, 0, 160, 64))
                result = self.extractor.extract_features(prepared)
                self.assertIsInstance(result.quality, FaceQuality)
                self.assertEqual(getattr(result.quality, fraction), 1)
                self.assertEqual(result.quality.laplacian_variance, 0)
                self.assertEqual(result.quality.supported_landmarks, 468)
                self.assertEqual(result.quality.clipped_landmarks, 0)
                self.assertEqual(result.quality.max_expression_score, 0.9)
                self.assertLess(result.quality.native_short_side, 256)
                for reason in (
                    warning,
                    "face_too_small",
                    "face_blurred_or_low_detail",
                    "non_neutral_expression",
                ):
                    self.assertIn(reason, result.quality.reasons)
                self.assertEqual(
                    json.loads(str(result.quality)), result.quality.to_dict()
                )
                self.assertEqual(
                    json.loads(str(result))["quality"], result.quality.to_dict()
                )
        self.extractor.quality_policy = replace(
            self.extractor.quality_policy, max_neutral_expression=0.95
        )
        adjusted = self.extractor.extract_features(prepared)
        self.assertNotIn("non_neutral_expression", adjusted.quality.reasons)
        self.assertEqual(adjusted.quality.max_expression_score, 0.9)

    def test_configured_support_count_changes_warning_without_losing_landmarks(self):
        """Apply feature-extraction support thresholds while keeping their measured count unchanged.
        Preserve raw predictions even when quality annotation requests further review.
        """
        detection = dense_prediction()
        for point in detection["faces"][0][100:]:
            point[1] = 0.75
        self.landmarker.detect.side_effect = None
        self.landmarker.detect.return_value = detection
        result = self.extract_image(self.image)
        self.assertEqual(result.quality.supported_landmarks, 100)
        self.assertIn("insufficient_head_support", result.quality.reasons)
        self.extractor.quality_policy = replace(
            self.extractor.quality_policy, min_supported_landmarks=50
        )
        adjusted = self.extract_image(self.image)
        self.assertNotIn("insufficient_head_support", adjusted.quality.reasons)
        self.assertEqual(result.landmarks, adjusted.landmarks)

    def test_scaled_transform_produces_orientation_during_extraction_without_angle_penalty(
        self,
    ):
        """Interpret positive uniform scale and large yaw/pitch/roll in the extraction result itself.
        Retain raw transforms and uncalibrated angles without requiring grouping or adding pose penalties.
        """
        self.landmarker.detect.side_effect = None
        for scale in (0.25, 1, 3.5):
            with self.subTest(scale=scale):
                detection = dense_prediction()
                detection["transforms"] = [face_transform(-75, 35, 70, scale)]
                self.landmarker.detect.return_value = detection
                result = self.extract_image(self.image)
                for name, angle in (("yaw", -75), ("pitch", 35), ("roll", 70)):
                    self.assertAlmostEqual(result.orientation[name + "_deg"], angle)
                self.assertFalse(result.orientation["calibrated"])
                self.assertEqual(result.facial_transform, detection["transforms"][0])
                self.assertNotIn("extreme_pose", result.quality.reasons)

    def test_invalid_or_missing_transforms_report_orientation_diagnostics_without_extraction_failure(
        self,
    ):
        """Treat missing, reflected, sheared and malformed transforms as unavailable orientation.
        Keep facial evidence and successful extraction independent of pose availability.
        """
        shear, perspective = np.eye(4), np.eye(4)
        shear[0, 1], perspective[3, 0] = 0.3, 1
        transforms = [
            None,
            "invalid",
            [[1]],
            np.zeros((4, 4)).tolist(),
            np.diag([1, 2, 1, 1]).tolist(),
            np.diag([-1, 1, 1, 1]).tolist(),
            np.full((4, 4), np.nan).tolist(),
            shear.tolist(),
            perspective.tolist(),
        ]
        self.landmarker.detect.side_effect = None
        for transform in transforms:
            with self.subTest(transform=transform):
                detection = dense_prediction()
                detection["transforms"] = [] if transform is None else [transform]
                self.landmarker.detect.return_value = detection
                result = next(
                    self.extractor.extract_features_batch(
                        [self.prepared_image(self.image)]
                    )
                )
                self.assertIsNone(result.error_type)
                self.assertEqual(
                    result.features.orientation["reason"], "missing_or_invalid_pose"
                )
                self.assertIsNotNone(result.features.landmarks)
                self.assertIsInstance(result.features.quality, FaceQuality)

    def test_quality_configuration_loads_independently_of_grouping_and_rejects_invalid_thresholds(
        self,
    ):
        """Load extraction-only warning thresholds without identity, grouping or pose-limit settings.
        Reject opacity/count/score domains that would make assessment invalid.
        """
        path = self.root / "quality.json"
        path.write_text(
            '{"min_face_size": 10, "min_landmark_alpha": 220}', encoding="utf-8"
        )
        policy = load_face_quality_policy(path)
        with FaceFeatureExtractor(device="cpu", quality_policy=policy) as extractor:
            self.assertIs(extractor.quality_policy, policy)
        for changes in (
            {"min_landmark_alpha": 0},
            {"min_landmark_alpha": 256},
            {"min_landmark_alpha": 1.5},
            {"min_supported_landmarks": 0},
            {"min_supported_landmarks": 469},
            {"max_neutral_expression": 1.1},
            {"max_neutral_expression": float("nan")},
            {"max_dark_fraction": True},
            {"min_laplacian_variance": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FaceQualityPolicy(**changes)

    def test_non_windows_cuda_requests_gpu_landmarks_without_retry(self):
        """Request the MediaPipe GPU delegate outside Windows when CUDA mode is selected.
        Propagate an unavailable delegate rather than constructing a second CPU detector.
        """
        with patch(
            "ailias.imageprocessing.face_feature_extraction.platform.system",
            return_value="Linux",
        ):
            processor = FaceFeatureExtractor(
                mediapipe_model=self.root / "mediapipe.model"
            )
        self.create_landmarker.side_effect = RuntimeError(
            "fixture GPU delegate unavailable"
        )
        with (
            processor,
            self.assertRaisesRegex(RuntimeError, "GPU delegate unavailable"),
        ):
            processor.detect_face_landmarks(self.image)
        self.create_landmarker.assert_called_once_with(
            self.root / "mediapipe.model", device="gpu"
        )


class DenseDelegateTests(unittest.TestCase):
    def test_dense_adapter_preserves_cpu_default_and_accepts_explicit_gpu(self):
        """Pass the selected delegate into MediaPipe while keeping legacy callers on CPU.
        Use a fake SDK so no model bundle, device or hardware context is created.
        """
        from ailias.imageprocessing.dense import DenseDetector

        base_options = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        base_options.Delegate = SimpleNamespace(CPU="cpu-fixture", GPU="gpu-fixture")
        create = Mock(return_value=Mock())
        sdk = SimpleNamespace(
            tasks=SimpleNamespace(
                BaseOptions=base_options,
                vision=SimpleNamespace(
                    FaceLandmarker=SimpleNamespace(create_from_options=create),
                    FaceLandmarkerOptions=Mock(
                        side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
                    ),
                ),
            )
        )
        with patch.dict(sys.modules, {"mediapipe": sdk}):
            cpu = DenseDetector("fixture.task")
            gpu = DenseDetector("fixture.task", device="gpu")
        self.assertEqual(
            create.call_args_list[0].args[0].base_options.delegate, "cpu-fixture"
        )
        self.assertEqual(
            create.call_args_list[1].args[0].base_options.delegate, "gpu-fixture"
        )
        cpu.close()
        gpu.close()


from dataclasses import replace
import json
