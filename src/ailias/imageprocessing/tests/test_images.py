from fractions import Fraction
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from PIL import Image

from ailias.imageprocessing.images import Frame, ImageProcessor
from ailias.imageprocessing.settings import ProcessingPolicy


def region_scores():
    """Provide deterministic BiSeNet scores with skin, hair and excluded clothing.
    Keep the shape real while replacing all learned inference with synthetic regions.
    """
    scores = np.zeros((1, 19, 512, 512), dtype=np.float32)
    scores[:, 1, :256] = 2
    scores[:, 17, :32] = 3
    scores[:, 16, 256:] = 2
    return scores


def dense_prediction():
    """Provide finite normalized landmarks and neutral expression/transform evidence.
    Distribute points over the synthetic skin region without claiming an anatomical fixture.
    """
    indices = np.arange(478)
    points = np.column_stack((0.2 + indices % 23 / 22 * 0.6,
                              0.1 + indices // 23 / 21 * 0.3, np.zeros(478)))
    return {"faces": [points.tolist()], "blendshapes": [{"mouthSmileLeft": 0.1}],
            "transforms": [np.eye(4).tolist()]}


class ImageProcessorTests(unittest.TestCase):
    def setUp(self):
        """Create explicit dummy model paths and deterministic fake network responses.
        Patch model constructors so units never load weights or execute a neural runtime.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        paths = {name: self.root / (name + ".model") for name in ("modnet", "bisenet", "mediapipe")}
        for path in paths.values():
            path.write_bytes(b"unit fixture, not model weights")
        self.modnet, self.bisenet = Mock(), Mock()
        self.modnet.forward.return_value = np.ones((1, 1, 512, 512), np.float32)
        self.bisenet.forward.side_effect = region_scores
        networks = {"modnet.model": self.modnet, "bisenet.model": self.bisenet}
        self.load = self.enterContext(patch(
            "cv2.dnn.readNetFromONNX", side_effect=lambda path: networks[Path(path).name]))
        self.landmarker = Mock()
        self.landmarker.detect.side_effect = lambda rgb: dense_prediction()
        self.create_landmarker = self.enterContext(patch(
            "ailias.imageprocessing.dense.DenseDetector", return_value=self.landmarker))
        self.processor = ImageProcessor(
            modnet_model=paths["modnet"], bisenet_model=paths["bisenet"],
            mediapipe_model=paths["mediapipe"], device="cpu",
            quality_policy=ProcessingPolicy(min_face_size=1, min_laplacian_variance=0.01))
        self.addCleanup(self.processor.close)
        y, x = np.indices((128, 160))
        values = ((x * 41 + y * 31) % 140 + 50).astype(np.uint8)
        self.image = Image.fromarray(values).convert("RGB")

    def test_pipeline_preserves_pixels_coordinates_and_returns_original_colour_features(self):
        """Exercise the full new path through fake MODNet, BiSeNet and MediaPipe outputs.
        Verify native-grid transparency, usable features, unchanged input and no retained sidecars.
        """
        before, files = self.image.tobytes(), set(self.root.iterdir())
        events = []
        self.modnet.setInput.side_effect = lambda blob: events.append("modnet")
        self.bisenet.setInput.side_effect = lambda blob: events.append("bisenet")
        self.landmarker.detect.side_effect = lambda rgb: events.append("mediapipe") or dense_prediction()
        result = self.processor.process_image(self.image, normalize_colour=False)
        self.assertEqual(events, ["modnet", "bisenet", "mediapipe"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["image"].size, self.image.size)
        self.assertEqual(result["image"].getpixel((80, 110))[3], 0)
        self.assertEqual(result["image"].getpixel((80, 25))[3], 255)
        np.testing.assert_array_equal(np.asarray(result["image"])[:, :, :3], np.asarray(self.image))
        normalized = np.asarray(result["landmarks"]["normalized_predicted"])
        np.testing.assert_allclose(result["landmarks"]["source_xy"], normalized[:, :2] * self.image.size)
        self.assertEqual(result["features"]["regions"]["skin"]["status"], "observed")
        self.assertEqual(result["features_colour_source"], "original_rgb")
        self.assertFalse(result["metric_depth_measured"])
        self.assertEqual(self.image.tobytes(), before)
        self.assertEqual(set(self.root.iterdir()), files)

    def test_models_are_lazy_reused_and_closed_after_collection(self):
        """Stream two inputs through the same cached models rather than rebuilding per photo.
        Verify explicit close releases the landmarker and prevents further inference.
        """
        self.load.assert_not_called()
        self.create_landmarker.assert_not_called()
        results = self.processor.process_images([self.image, self.image], normalize_colour=False)
        self.load.assert_not_called()
        self.assertEqual(next(results)["status"], "ready")
        self.assertEqual(next(results)["status"], "ready")
        self.assertEqual(self.load.call_count, 2)
        self.create_landmarker.assert_called_once()
        self.processor.close()
        self.processor.close()
        self.landmarker.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.processor.process_image(self.image)

    def test_partial_alpha_survives_matting_and_semantic_exclusion(self):
        """Keep fractional MODNet opacity and prior transparency in retained head regions.
        Ensure semantic clothing/background exclusion never restores a removed pixel.
        """
        self.modnet.forward.return_value.fill(0.5)
        image = self.image.convert("RGBA")
        image.putalpha(128)
        image.putpixel((0, 0), (200, 30, 80, 0))
        matte = self.processor.remove_background(image)
        head, labels = self.processor.isolate_head(matte)
        self.assertEqual(head.getpixel((80, 25))[3], 64)
        self.assertEqual(head.getpixel((80, 110))[3], 0)
        self.assertEqual(head.getpixel((0, 0))[3], 0)
        self.assertEqual(labels.getpixel((0, 0)), 0)
        self.assertEqual(labels.getpixel((80, 110)), 16)

    def test_transparent_rgb_is_hidden_from_bisenet_and_mediapipe(self):
        """Composite transparent pixels over neutral grey before either face model sees RGB.
        Prevent hidden source background colours from leaking into model inputs.
        """
        image = Image.new("RGBA", (128, 128), (250, 0, 0, 0))
        self.processor.isolate_head(image)
        blob = self.bisenet.setInput.call_args.args[0]
        expected = (np.array([127, 127, 127]) / 255 - [.485, .456, .406]) / [.229, .224, .225]
        np.testing.assert_allclose(blob[0, :, 100, 100], expected, atol=1e-6)
        self.processor.detect_face_landmarks(image)
        rgb = self.landmarker.detect.call_args.args[0]
        np.testing.assert_array_equal(rgb, np.full((128, 128, 3), 127, np.uint8))

    def test_no_face_multiple_faces_and_invalid_points_abstain_without_features(self):
        """Require one valid dense face rather than silently selecting a subject.
        Keep the prepared cutout available while returning explicit review reasons.
        """
        for faces, reason in (
            ([], "no_face_detected"),
            (dense_prediction()["faces"] * 2, "multiple_faces_detected"),
            ([[[float("nan"), 0, 0]] * 478], "invalid_dense_landmarks"),
        ):
            with self.subTest(reason=reason):
                self.landmarker.detect.side_effect = None
                self.landmarker.detect.return_value = {"faces": faces, "blendshapes": [], "transforms": []}
                result = self.processor.process_image(self.image, normalize_colour=False)
                self.assertEqual(result["status"], "review")
                self.assertIn(reason, result["quality"]["reasons"])
                self.assertIsNone(result["features"])

    def test_empty_head_skips_landmark_inference(self):
        """Abstain when semantic selection removes all foreground from a portrait.
        Avoid loading MediaPipe for an image with no supported head pixels.
        """
        scores = np.zeros((1, 19, 512, 512), np.float32)
        scores[:, 16] = 1
        self.bisenet.forward.side_effect = None
        self.bisenet.forward.return_value = scores
        result = self.processor.process_image(self.image)
        self.assertEqual(result["quality"]["reasons"], ["no_head_foreground"])
        self.create_landmarker.assert_not_called()

    def test_quality_flags_low_detail_and_expression_without_discarding_evidence(self):
        """Expose low-detail and non-neutral-expression warnings for downstream review.
        Still return extracted evidence instead of treating successful inference as approval.
        """
        detection = dense_prediction()
        detection["blendshapes"][0]["mouthSmileLeft"] = .9
        self.landmarker.detect.side_effect = None
        self.landmarker.detect.return_value = detection
        result = self.processor.process_image(Image.new("RGB", self.image.size, (160, 110, 75)),
                                              normalize_colour=False)
        self.assertEqual(result["status"], "review")
        self.assertIn("face_blurred_or_low_detail", result["quality"]["reasons"])
        self.assertIn("non_neutral_expression", result["quality"]["reasons"])
        self.assertIsNotNone(result["features"])

    def test_malformed_model_predictions_fail_instead_of_publishing_masks(self):
        """Reject wrong-sized or nonfinite network outputs before creating cutouts.
        Exercise failures from both the MODNet opacity and BiSeNet class-score contracts.
        """
        for prediction in (np.zeros((1, 1, 2, 2)), np.full((1, 1, 512, 512), np.nan)):
            self.modnet.forward.return_value = prediction
            with self.assertRaisesRegex(ValueError, "MODNet"):
                self.processor.remove_background(self.image)
        self.bisenet.forward.side_effect = None
        for prediction in (np.zeros((1, 18, 512, 512)), np.full((1, 19, 512, 512), np.nan)):
            self.bisenet.forward.return_value = prediction
            with self.assertRaisesRegex(ValueError, "BiSeNet"):
                self.processor.isolate_head(self.image)

    def test_quality_flags_extreme_or_unavailable_pose_without_claiming_calibration(self):
        """Flag a strongly rolled face and an invalid transform for explicit review.
        Preserve the predicted pose as uncalibrated image evidence rather than measured orientation.
        """
        angle = np.deg2rad(40)
        transform = np.eye(4)
        transform[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        for transforms, reason in (([transform.tolist()], "extreme_pose"),
                                    ([], "missing_or_invalid_pose")):
            with self.subTest(reason=reason):
                detection = dense_prediction()
                detection["transforms"] = transforms
                self.landmarker.detect.side_effect = None
                self.landmarker.detect.return_value = detection
                result = self.processor.process_image(self.image, normalize_colour=False)
                self.assertIn(reason, result["quality"]["reasons"])
                if transforms:
                    self.assertAlmostEqual(result["quality"]["pose"]["roll_deg"], 40)
                    self.assertFalse(result["quality"]["pose"]["calibrated"])

    def test_configured_alpha_threshold_controls_foreground_support_review(self):
        """Allow the configured opacity threshold to decide whether points have head support.
        Preserve the same fractional cutout rather than modifying alpha to satisfy the quality gate.
        """
        self.modnet.forward.return_value.fill(.9)
        original = self.processor.process_image(self.image, normalize_colour=False)
        self.assertIn("insufficient_head_support", original["quality"]["reasons"])
        self.processor.quality_policy = replace(self.processor.quality_policy, min_landmark_alpha=220)
        adjusted = self.processor.process_image(self.image, normalize_colour=False)
        self.assertNotIn("insufficient_head_support", adjusted["quality"]["reasons"])
        self.assertEqual(original["image"].tobytes(), adjusted["image"].tobytes())

    def test_configured_supported_point_count_controls_review_without_dropping_features(self):
        """Apply the requested minimum number of supported base-face landmarks.
        Keep regional evidence available when a stricter support requirement requests review.
        """
        detection = dense_prediction()
        for point in detection["faces"][0][100:]:
            point[1] = .75
        self.landmarker.detect.side_effect = None
        self.landmarker.detect.return_value = detection
        original = self.processor.process_image(self.image, normalize_colour=False)
        self.assertIn("insufficient_head_support", original["quality"]["reasons"])
        self.assertIsNotNone(original["features"])
        self.processor.quality_policy = replace(self.processor.quality_policy, min_supported_landmarks=50)
        adjusted = self.processor.process_image(self.image, normalize_colour=False)
        self.assertNotIn("insufficient_head_support", adjusted["quality"]["reasons"])

    def test_configured_expression_and_pose_limits_control_review_flags(self):
        """Read expression, pitch/roll and yaw limits from the supplied quality policy.
        Changing a review threshold must not change the underlying predicted transform or scores.
        """
        cases = (
            ("non_neutral_expression", {"max_neutral_expression": .95}, "expression", .9),
            ("extreme_pose", {"max_pitch_roll_degrees": 45}, "roll", 40),
            ("extreme_pose", {"max_yaw_degrees": 85}, "yaw", 80),
        )
        base_policy = self.processor.quality_policy
        for reason, changes, kind, value in cases:
            with self.subTest(kind=kind):
                detection = dense_prediction()
                if kind == "expression":
                    detection["blendshapes"][0]["mouthSmileLeft"] = value
                else:
                    sine, cosine = np.sin(np.deg2rad(value)), np.cos(np.deg2rad(value))
                    matrix = np.eye(4)
                    axes = (0, 1) if kind == "roll" else (0, 2)
                    matrix[np.ix_(axes, axes)] = [[cosine, -sine], [sine, cosine]]
                    detection["transforms"] = [matrix.tolist()]
                self.landmarker.detect.side_effect = None
                self.landmarker.detect.return_value = detection
                self.processor.quality_policy = base_policy
                original = self.processor.process_image(self.image, normalize_colour=False)
                self.assertIn(reason, original["quality"]["reasons"])
                self.processor.quality_policy = replace(base_policy, **changes)
                adjusted = self.processor.process_image(self.image, normalize_colour=False)
                self.assertNotIn(reason, adjusted["quality"]["reasons"])
                self.assertEqual(original["facial_transform"], adjusted["facial_transform"])

    def test_quality_policy_rejects_review_thresholds_outside_their_meaningful_ranges(self):
        """Reject opacity/count/score/angle thresholds outside their documented domains.
        Prevent invalid configuration from weakening or breaking face-quality assessment.
        """
        for changes in (
            {"min_landmark_alpha": 0}, {"min_landmark_alpha": 256}, {"min_landmark_alpha": 1.5},
            {"min_supported_landmarks": 0}, {"min_supported_landmarks": 469},
            {"max_neutral_expression": 1.1}, {"max_neutral_expression": float("nan")},
            {"max_pitch_roll_degrees": 0}, {"max_pitch_roll_degrees": 181},
            {"max_yaw_degrees": 0}, {"max_yaw_degrees": 91},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.processor.quality_policy, **changes)

    def test_outside_and_clipped_landmarks_preserve_distinct_review_outcomes(self):
        """Preserve early rejection for an out-of-image face and a warning for partial clipping.
        Return features only when the predicted face still overlaps the source image.
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
                result = self.processor.process_image(self.image, normalize_colour=False)
                reason = "face_outside_image" if outside else "face_clipped"
                self.assertIn(reason, result["quality"]["reasons"])
                self.assertEqual(result["features"] is None, outside)

    def test_batch_continues_after_bad_file_and_preserves_frame_metadata(self):
        """Return a per-item error for unreadable files without losing later inputs.
        Carry video presentation time and original decode index through feature processing.
        """
        frame = Frame(self.image, self.root / "video.mp4", 12, 1.5)
        results = list(self.processor.process_images([self.root / "missing.png", frame],
                                                     normalize_colour=False))
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["error_type"], "FileNotFoundError")
        self.assertEqual(results[1]["status"], "ready")
        self.assertEqual(results[1]["source"], frame.source)
        self.assertEqual(results[1]["frame_index"], 12)
        self.assertEqual(results[1]["timestamp_seconds"], 1.5)

    def test_collection_reports_inference_failures_and_reuses_model_on_next_input(self):
        """Report OpenCV inference failures per input rather than stopping the collection.
        Reuse the same cached network when the next independent image can be processed.
        """
        import cv2

        self.modnet.forward.side_effect = [
            cv2.error("fixture inference error"), np.ones((1, 1, 512, 512), np.float32)]
        results = list(self.processor.process_images([self.image, self.image],
                                                     normalize_colour=False))
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["reason"], "MODNet inference failed")
        self.assertEqual(results[1]["status"], "ready")
        self.assertEqual(self.load.call_count, 2)

    def test_colour_normalization_uses_scene_neutrals_and_preserves_alpha(self):
        """Correct a supported scene cast while excluding the supplied head region.
        Preserve transparency and abstain rather than forcing correction on saturated scenes.
        """
        image = Image.new("RGBA", (128, 128), (110, 100, 100, 128))
        image.paste((200, 80, 40, 128), (40, 40, 88, 88))
        corrected, record = self.processor.normalize_colour(image, excluded_bounds=(40, 40, 88, 88))
        self.assertEqual(record["status"], "estimated_scene_neutrals")
        pixel = corrected.getpixel((0, 0))
        self.assertLessEqual(max(pixel[:3]) - min(pixel[:3]), 1)
        self.assertEqual(pixel[3], 128)
        unchanged, skipped = self.processor.normalize_colour(Image.new("RGB", (128, 128), "red"))
        self.assertTrue(skipped["status"].startswith("skipped_"))
        self.assertEqual(unchanged.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(image.getpixel((0, 0)), (110, 100, 100, 128))

    def test_colour_changes_cutout_not_original_feature_samples(self):
        """Keep feature colour evidence tied to original RGB even when output is normalized.
        Return correction provenance separately so callers cannot confuse the two pixel sources.
        """
        self.processor.normalize_colour = Mock(return_value=(
            Image.new("RGB", self.image.size, "blue"), {"status": "fixture_correction"}))
        result = self.processor.process_image(self.image)
        self.assertEqual(result["image"].getpixel((80, 25))[:3], (0, 0, 255))
        self.assertEqual(result["colour"]["status"], "fixture_correction")
        skin = result["features"]["regions"]["skin"]["median_linear_rgb"]
        self.assertAlmostEqual(skin[0], skin[1])
        self.assertAlmostEqual(skin[1], skin[2])

    def test_explicit_cropping_preserves_pixels_and_rejects_padding(self):
        """Crop caller-selected native pixels without an automatic face detector.
        Reject padding, inverted bounds and fractional coordinate contracts.
        """
        result = self.processor.crop_image(self.image, (10, 20, 40, 50))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(self.image)[20:50, 10:40])
        for bounds in ((-1, 0, 5, 5), (0, 0, 1000, 5), (5, 5, 2, 8), (0.5, 0, 5, 5)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                self.processor.crop_image(self.image, bounds)

    def test_loading_preserves_transparency_without_changing_legacy_rgb_decoding(self):
        """Retain PNG transparency for the new API without changing existing decoder defaults.
        Detach loaded pixels from their file so callers can safely process them after loading.
        """
        from ailias.imageprocessing.image_io import read_oriented_image

        source = self.root / "transparent.png"
        image = self.image.convert("RGBA")
        image.putalpha(128)
        image.save(source)
        loaded = self.processor.load_image(source)
        self.assertEqual(loaded.mode, "RGBA")
        self.assertEqual(loaded.getpixel((0, 0))[3], 128)
        legacy, metadata = read_oriented_image(source)
        self.assertEqual(legacy.mode, "RGB")
        self.assertNotIn("alpha_preserved", metadata)
        source.unlink()
        self.assertEqual(loaded.getpixel((0, 0)), image.getpixel((0, 0)))

    def test_missing_model_and_exception_cleanup_do_not_load_or_download_weights(self):
        """Fail on missing explicit models and release owned resources after exceptions.
        Keep lightweight image loading/cropping independent of model configuration.
        """
        with ImageProcessor(device="cpu") as processor:
            with self.assertRaisesRegex(ValueError, "modnet"):
                processor.remove_background(self.image)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with self.processor:
                self.processor.detect_face_landmarks(self.image)
                raise RuntimeError("fixture")
        self.landmarker.close.assert_called_once()

    def test_default_cuda_uses_gpu_networks_and_explicit_windows_landmarker_exception(self):
        """Default CUDA mode keeps MODNet/BiSeNet on GPU but selects MediaPipe CPU on Windows.
        Verify graph evidence and delegate selection using mocks, without probing GPU hardware.
        """
        import cv2

        self.modnet.dump.return_value = 'layer [label="conv CUDA/CUDA"];'
        self.bisenet.dump.return_value = 'layer [label="conv CUDA/CUDA"];'
        with patch("ailias.imageprocessing.images.platform.system", return_value="Windows"):
            processor = ImageProcessor(
                modnet_model=self.root / "modnet.model", bisenet_model=self.root / "bisenet.model",
                mediapipe_model=self.root / "mediapipe.model", quality_policy=self.processor.quality_policy)
        with processor, patch("ailias.imageprocessing.cuda.require_cuda") as require_cuda, patch(
            "cv2.cuda.setDevice"
        ) as select_device:
            result = processor.process_image(self.image, normalize_colour=False)
        self.assertEqual(require_cuda.call_count, 2)
        self.assertEqual(select_device.call_count, 2)
        self.modnet.setPreferableBackend.assert_called_once_with(cv2.dnn.DNN_BACKEND_CUDA)
        self.bisenet.setPreferableTarget.assert_called_once_with(cv2.dnn.DNN_TARGET_CUDA)
        self.create_landmarker.assert_called_once_with(self.root / "mediapipe.model", device="cpu")
        self.assertEqual(result["execution"]["requested_device"], "cuda")
        self.assertEqual(result["execution"]["networks"]["modnet"]["status"], "verified")
        self.assertEqual(result["execution"]["mediapipe_device"], "cpu")
        self.assertTrue(result["execution"]["mediapipe_executed"])

    def test_non_windows_cuda_requests_gpu_landmarks_without_retry(self):
        """Request the MediaPipe GPU delegate outside Windows when CUDA mode is selected.
        Propagate an unavailable delegate rather than constructing a second CPU detector.
        """
        with patch("ailias.imageprocessing.images.platform.system", return_value="Linux"):
            processor = ImageProcessor(mediapipe_model=self.root / "mediapipe.model")
        self.create_landmarker.side_effect = RuntimeError("fixture GPU delegate unavailable")
        with processor, self.assertRaisesRegex(RuntimeError, "GPU delegate unavailable"):
            processor.detect_face_landmarks(self.image)
        self.create_landmarker.assert_called_once_with(self.root / "mediapipe.model", device="gpu")

    def test_explicit_cpu_never_checks_or_selects_cuda(self):
        """Keep all inference on CPU only when the caller explicitly requests CPU mode.
        Do not inspect GPU availability or silently change the selected hardware.
        """
        import cv2

        with patch("ailias.imageprocessing.cuda.require_cuda") as require_cuda, patch(
            "cv2.cuda.setDevice"
        ) as select_device:
            result = self.processor.process_image(self.image, normalize_colour=False)
        require_cuda.assert_not_called()
        select_device.assert_not_called()
        self.modnet.setPreferableBackend.assert_called_once_with(cv2.dnn.DNN_BACKEND_OPENCV)
        self.bisenet.setPreferableTarget.assert_called_once_with(cv2.dnn.DNN_TARGET_CPU)
        self.create_landmarker.assert_called_once_with(self.root / "mediapipe.model", device="cpu")
        self.assertEqual(result["execution"]["requested_device"], "cpu")

    def test_unavailable_cuda_never_loads_a_cpu_network(self):
        """Fail the requested CUDA operation before loading a model when CUDA is unavailable.
        Keep the caller in control of CPU mode rather than retrying through another backend.
        """
        from ailias.imageprocessing.cuda import CudaUnavailable

        with ImageProcessor(modnet_model=self.root / "modnet.model") as processor, patch(
            "ailias.imageprocessing.cuda.require_cuda", side_effect=CudaUnavailable("fixture CUDA unavailable")
        ), self.assertRaises(CudaUnavailable):
            processor.remove_background(self.image)
        self.load.assert_not_called()

    def test_cuda_graph_without_cuda_layers_is_rejected(self):
        """Refuse an OpenCV forward that completed entirely on CPU despite requesting CUDA.
        Check actual graph evidence rather than treating backend configuration as GPU execution.
        """
        from ailias.imageprocessing.cuda import CudaUnavailable

        self.modnet.dump.return_value = 'layer [label="conv OCV/CPU"];'
        with ImageProcessor(modnet_model=self.root / "modnet.model") as processor, patch(
            "ailias.imageprocessing.cuda.require_cuda"
        ), patch("cv2.cuda.setDevice"), self.assertRaises(CudaUnavailable):
            processor.remove_background(self.image)
        self.assertEqual(self.load.call_count, 1)

    def test_invalid_device_selection_fails_before_loading_models(self):
        """Validate backend names and GPU indices without importing an inference runtime.
        Avoid ambiguous devices, negative GPU indices and boolean-as-integer configuration.
        """
        for options in ({"device": "auto"}, {"gpu_index": -1}, {"gpu_index": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ImageProcessor(**options)
        self.load.assert_not_called()


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


class VideoFrameTests(unittest.TestCase):
    def container(self, timestamps):
        """Build a fake timestamped video container with no real decoder or media file.
        Convert selected frames to tiny RGB arrays only when sampling requests pixels.
        """
        frames = [SimpleNamespace(
            pts=pts, time_base=Fraction(1, 10), rotation=0,
            to_ndarray=Mock(return_value=np.full((4, 6, 3), index, np.uint8)),
        ) for index, pts in enumerate(timestamps)]
        container = MagicMock()
        container.__enter__.return_value = container
        container.streams.video = [SimpleNamespace(metadata={})]
        container.decode.return_value = iter(frames)
        return container, frames

    def test_sampling_uses_pts_and_closes_early_without_decoding_extra_frames(self):
        """Sample variable-rate input by elapsed presentation time rather than frame count.
        Respect the requested limit and release the decoder before another frame is read.
        """
        container, frames = self.container([100, 102, 108, 115])
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=Mock(return_value=container))}):
            output = list(ImageProcessor.extract_video_frames("fixture.mp4", interval_seconds=.5,
                                                               max_frames=2))
        self.assertEqual([frame.index for frame in output], [0, 2])
        np.testing.assert_allclose([frame.timestamp_seconds for frame in output], [0, .8])
        frames[1].to_ndarray.assert_not_called()
        frames[3].to_ndarray.assert_not_called()
        container.__exit__.assert_called_once()

    def test_decoder_is_released_when_consumer_closes_generator(self):
        """Close the video container when a streaming consumer stops before exhaustion.
        Avoid keeping file handles or decoder resources alive after an explicit iterator close.
        """
        container, _ = self.container([0, 10, 20])
        with patch.dict(sys.modules, {"av": SimpleNamespace(open=Mock(return_value=container))}):
            iterator = ImageProcessor.extract_video_frames("fixture.mp4")
            next(iterator)
            iterator.close()
        container.__exit__.assert_called_once()

    def test_missing_or_nonmonotonic_timestamps_and_rotation_are_rejected(self):
        """Reject inputs whose timing or orientation cannot meet the frame-coordinate contract.
        Release the decoder on every failure without guessing FPS or silently rotating pixels.
        """
        for timestamps, rotation in (([None], 0), ([10, 10], 0), ([10], 90)):
            with self.subTest(timestamps=timestamps, rotation=rotation):
                container, frames = self.container(timestamps)
                frames[0].rotation = rotation
                with patch.dict(sys.modules, {"av": SimpleNamespace(open=Mock(return_value=container))}):
                    with self.assertRaises(ValueError):
                        list(ImageProcessor.extract_video_frames("fixture.mp4", interval_seconds=.1))
                container.__exit__.assert_called_once()

    def test_invalid_sampling_options_are_rejected_before_decoder_import(self):
        """Validate sampling limits before opening a file or importing optional decoding tools.
        Prevent zero/negative/nonfinite intervals and nonpositive frame limits.
        """
        for options in ({"interval_seconds": 0}, {"interval_seconds": float("nan")},
                        {"interval_seconds": True}, {"max_frames": 0}, {"max_frames": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                next(ImageProcessor.extract_video_frames("fixture.mp4", **options))
