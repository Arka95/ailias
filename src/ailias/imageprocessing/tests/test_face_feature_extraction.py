from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ailias.imageprocessing.face_feature_extraction import (
    FaceExtractionResult, FaceFeatureExtractor, FaceFeatures, FaceFeaturesBiseNet,
)


def dense_prediction():
    """Provide finite normalized landmarks and neutral expression/transform evidence.
    Distribute points over synthetic skin without claiming an anatomical fixture.
    """
    indices = np.arange(478)
    points = np.column_stack((0.2 + indices % 23 / 22 * 0.6,
                              0.1 + indices // 23 / 21 * 0.3, np.zeros(478)))
    return {"faces": [points.tolist()], "blendshapes": [{"mouthSmileLeft": 0.1}],
            "transforms": [np.eye(4).tolist()]}


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
        self.create_landmarker = self.enterContext(patch(
            "ailias.imageprocessing.dense.DenseDetector", return_value=self.landmarker))
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
        labels[:image.height // 2] = 1
        labels[:image.height // 16] = 17
        head = image.convert("RGBA")
        head.putalpha(Image.fromarray(np.where(labels == 16, 0, 255).astype(np.uint8)))
        return {"image": head, "original_image": image.copy(), "regions": Image.fromarray(labels),
                "source": self.root / "video.mp4", "frame_index": 12, "timestamp_seconds": 1.5}

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
        before = {key: prepared[key].tobytes() for key in ("image", "original_image", "regions")}
        files = set(self.root.iterdir())
        with patch("cv2.dnn.readNetFromONNX", side_effect=AssertionError("Preparation was rerun")):
            result = self.extractor.extract_features(prepared)
        self.assertIsInstance(result, FaceFeatures)
        normalized = np.asarray(result.landmarks["normalized_predicted"])
        np.testing.assert_allclose(result.landmarks["source_xy"], normalized[:, :2] * self.image.size)
        skin = result.features["regions"]["skin"]
        self.assertIsInstance(result.bisenet, FaceFeaturesBiseNet)
        self.assertEqual(result.bisenet.skin.to_dict(), skin)
        self.assertEqual(skin["status"], "observed")
        self.assertAlmostEqual(skin["median_linear_rgb"][0], skin["median_linear_rgb"][2])
        self.assertEqual(result.features_colour_source, "original_rgb")
        self.assertFalse(result.features["metric_depth_measured"])
        for key, pixels in before.items():
            self.assertEqual(prepared[key].tobytes(), pixels)
        self.assertEqual(set(self.root.iterdir()), files)

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
        self.create_landmarker.assert_called_once_with(self.root / "mediapipe.model", device="cpu")
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
        self.extractor.detect_face_landmarks(Image.new("RGBA", (128, 128), (250, 0, 0, 0)))
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
            ("image", self.image), ("original_image", None), ("regions", self.image),
            ("original_image", Image.new("RGB", (10, 10))),
            ("regions", Image.new("L", (10, 10))),
        ):
            with self.subTest(field=field):
                prepared = {**self.prepared_image(self.image), field: value}
                with self.assertRaises(ValueError):
                    self.extractor.extract_features(prepared)
        self.create_landmarker.assert_not_called()

    def test_batch_forwards_preparation_errors_and_continues_after_extraction_failure(self):
        """Preserve preparation failures and isolate failed landmark inference to its input.
        Reuse the task for the next prepared item without rerunning any preparation stage.
        """
        prepared = self.prepared_image(self.image)
        error = {"status": "error", "error_type": "FileNotFoundError",
                 "reason": "fixture missing input", "source": "missing.png"}
        self.landmarker.detect.side_effect = [RuntimeError("fixture"), dense_prediction()]
        results = list(self.extractor.extract_features_batch([error, prepared, prepared]))
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

    def test_feature_visibility_records_positive_alpha_without_collection_quality_thresholds(self):
        """Retain faint visible pixels as raw support rather than applying a quality-selection threshold.
        Leave stricter opacity warnings to grouping while preserving the original head pixels.
        """
        prepared = self.prepared_image(self.image)
        alpha = np.asarray(prepared["image"].getchannel("A")).copy()
        alpha[alpha != 0] = 230
        prepared["image"].putalpha(Image.fromarray(alpha))
        before = prepared["image"].tobytes()
        result = self.extractor.extract_features(prepared)
        self.assertTrue(all(result.landmarks["on_head"]))
        self.assertEqual(prepared["image"].tobytes(), before)

    def test_default_cuda_keeps_explicit_windows_cpu_exception(self):
        """Select MediaPipe's CPU delegate on Windows even in the default CUDA mode.
        Verify the documented exception without probing hardware or silently retrying.
        """
        with patch("ailias.imageprocessing.face_feature_extraction.platform.system", return_value="Windows"):
            extractor = FaceFeatureExtractor(mediapipe_model=self.root / "mediapipe.model")
        with extractor:
            result = next(extractor.extract_features_batch([self.prepared_image(self.image)]))
        self.create_landmarker.assert_called_once_with(self.root / "mediapipe.model", device="cpu")
        self.assertEqual(result.execution, {
            "requested_device": "cuda", "mediapipe_device": "cpu", "mediapipe_executed": True})

    def test_missing_model_and_exception_cleanup_do_not_load_or_download_weights(self):
        """Fail on missing local weights and release an owned task after context exceptions.
        Constructing an extractor never starts inference, downloads models or launches processes.
        """
        with FaceFeatureExtractor(device="cpu") as extractor:
            with self.assertRaisesRegex(ValueError, "mediapipe"):
                extractor.extract_features(self.prepared_image(self.image))
        with FaceFeatureExtractor(mediapipe_model=self.root / "missing.task", device="cpu") as extractor:
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
                self.landmarker.detect.return_value = {"faces": faces, "blendshapes": [], "transforms": []}
                result = next(self.extractor.extract_features_batch([self.prepared_image(self.image)]))
                self.assertEqual(result.reason, reason)
                self.assertIsNone(result.features.features)
                self.assertIsInstance(result.features.bisenet, FaceFeaturesBiseNet)
                self.assertEqual(result.features.bisenet.skin.status, "observed")

    def test_outside_and_clipped_landmarks_preserve_distinct_review_outcomes(self):
        """Retain structurally valid point predictions even when outside or clipped against the image.
        Leave image-overlap validation and selection to the grouping stage.
        """
        for outside in (True, False):
            with self.subTest(outside=outside):
                detection = dense_prediction()
                if outside:
                    for point in detection["faces"][0]:
                        point[0] += 2
                else:
                    detection["faces"][0][0][0] = -.1
                self.landmarker.detect.side_effect = None
                self.landmarker.detect.return_value = detection
                result = self.extract_image(self.image)
                self.assertIsNotNone(result.features)
                self.assertEqual(result.landmarks["normalized_predicted"], detection["faces"][0])

    def test_non_windows_cuda_requests_gpu_landmarks_without_retry(self):
        """Request the MediaPipe GPU delegate outside Windows when CUDA mode is selected.
        Propagate an unavailable delegate rather than constructing a second CPU detector.
        """
        with patch("ailias.imageprocessing.face_feature_extraction.platform.system", return_value="Linux"):
            processor = FaceFeatureExtractor(mediapipe_model=self.root / "mediapipe.model")
        self.create_landmarker.side_effect = RuntimeError("fixture GPU delegate unavailable")
        with processor, self.assertRaisesRegex(RuntimeError, "GPU delegate unavailable"):
            processor.detect_face_landmarks(self.image)
        self.create_landmarker.assert_called_once_with(self.root / "mediapipe.model", device="gpu")


class DenseDelegateTests(unittest.TestCase):
    def test_dense_adapter_preserves_cpu_default_and_accepts_explicit_gpu(self):
        """Pass the selected delegate into MediaPipe while keeping legacy callers on CPU.
        Use a fake SDK so no model bundle, device or hardware context is created.
        """
        from ailias.imageprocessing.dense import DenseDetector

        base_options = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        base_options.Delegate = SimpleNamespace(CPU="cpu-fixture", GPU="gpu-fixture")
        create = Mock(return_value=Mock())
        sdk = SimpleNamespace(tasks=SimpleNamespace(
            BaseOptions=base_options, vision=SimpleNamespace(
                FaceLandmarker=SimpleNamespace(create_from_options=create),
                FaceLandmarkerOptions=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)))))
        with patch.dict(sys.modules, {"mediapipe": sdk}):
            cpu = DenseDetector("fixture.task")
            gpu = DenseDetector("fixture.task", device="gpu")
        self.assertEqual(create.call_args_list[0].args[0].base_options.delegate, "cpu-fixture")
        self.assertEqual(create.call_args_list[1].args[0].base_options.delegate, "gpu-fixture")
        cpu.close()
        gpu.close()
