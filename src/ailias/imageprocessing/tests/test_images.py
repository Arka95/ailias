from fractions import Fraction
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from PIL import Image

from ailias.imageprocessing.images import Frame, ImageProcessor


def region_scores():
    """Provide deterministic BiSeNet scores with skin, hair and excluded clothing.
    Keep the shape real while replacing all learned inference with synthetic regions.
    """
    scores = np.zeros((1, 19, 512, 512), dtype=np.float32)
    scores[:, 1, :256] = 2
    scores[:, 17, :32] = 3
    scores[:, 16, 256:] = 2
    return scores


class ImageProcessorTests(unittest.TestCase):
    def setUp(self):
        """Create explicit dummy model paths and deterministic fake network responses.
        Patch model constructors so units never load weights or execute a neural runtime.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        paths = {name: self.root / (name + ".model") for name in ("modnet", "bisenet")}
        for path in paths.values():
            path.write_bytes(b"unit fixture, not model weights")
        self.modnet, self.bisenet = Mock(), Mock()
        self.modnet.forward.return_value = np.ones((1, 1, 512, 512), np.float32)
        self.bisenet.forward.side_effect = region_scores
        networks = {"modnet.model": self.modnet, "bisenet.model": self.bisenet}
        self.load = self.enterContext(patch(
            "cv2.dnn.readNetFromONNX", side_effect=lambda path: networks[Path(path).name]))
        self.create_landmarker = self.enterContext(patch(
            "ailias.imageprocessing.dense.DenseDetector",
            side_effect=AssertionError("Image preparation must not construct MediaPipe")))
        self.processor = ImageProcessor(
            modnet_model=paths["modnet"], bisenet_model=paths["bisenet"],
            device="cpu")
        self.addCleanup(self.processor.close)
        y, x = np.indices((128, 160))
        values = ((x * 41 + y * 31) % 140 + 50).astype(np.uint8)
        self.image = Image.fromarray(values).convert("RGB")

    def test_preparation_preserves_pixels_and_labels_without_landmark_inference(self):
        """Prepare native-grid pixels with MODNet and BiSeNet but no MediaPipe model or dependency.
        Keep original RGB and semantic labels in memory without modifying inputs or writing sidecars.
        """
        before, files = self.image.tobytes(), set(self.root.iterdir())
        events = []
        self.modnet.setInput.side_effect = lambda blob: events.append("modnet")
        self.bisenet.setInput.side_effect = lambda blob: events.append("bisenet")
        with patch.dict(sys.modules, {"mediapipe": None}):
            result = self.processor.process_image(self.image, normalize_colour=False)
        self.assertEqual(events, ["modnet", "bisenet"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["image"].size, self.image.size)
        self.assertEqual(result["image"].getpixel((80, 110))[3], 0)
        self.assertEqual(result["image"].getpixel((80, 25))[3], 255)
        np.testing.assert_array_equal(np.asarray(result["image"])[:, :, :3], np.asarray(self.image))
        self.assertEqual(result["regions"].size, self.image.size)
        self.assertEqual(result["regions"].getpixel((80, 110)), 16)
        self.assertEqual(result["original_image"].tobytes(), before)
        self.create_landmarker.assert_not_called()
        self.assertEqual(self.image.tobytes(), before)
        self.assertEqual(set(self.root.iterdir()), files)

    def test_models_are_lazy_reused_and_closed_after_collection(self):
        """Stream two inputs through the same cached models rather than rebuilding per photo.
        Verify explicit close prevents further inference without owning a landmark task.
        """
        self.load.assert_not_called()
        self.create_landmarker.assert_not_called()
        results = self.processor.process_images([self.image, self.image], normalize_colour=False)
        self.load.assert_not_called()
        self.assertEqual(next(results)["status"], "ready")
        self.assertEqual(next(results)["status"], "ready")
        self.assertEqual(self.load.call_count, 2)
        self.create_landmarker.assert_not_called()
        self.processor.close()
        self.processor.close()
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

    def test_transparent_rgb_is_hidden_from_bisenet(self):
        """Composite transparent pixels over neutral grey before BiSeNet sees RGB.
        Prevent hidden source background colours from leaking into model inputs.
        """
        image = Image.new("RGBA", (128, 128), (250, 0, 0, 0))
        self.processor.isolate_head(image)
        blob = self.bisenet.setInput.call_args.args[0]
        expected = (np.array([127, 127, 127]) / 255 - [.485, .456, .406]) / [.229, .224, .225]
        np.testing.assert_allclose(blob[0, :, 100, 100], expected, atol=1e-6)

    def test_empty_head_requests_review_before_colour_normalization(self):
        """Abstain when semantic selection removes all foreground from a portrait.
        Do not estimate white balance or claim usable preparation for an empty head.
        """
        scores = np.zeros((1, 19, 512, 512), np.float32)
        scores[:, 16] = 1
        self.bisenet.forward.side_effect = None
        self.bisenet.forward.return_value = scores
        with patch.object(self.processor, "normalize_colour") as normalize_colour:
            result = self.processor.process_image(self.image)
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["quality"]["reasons"], ["no_head_foreground"])
        normalize_colour.assert_not_called()
        self.create_landmarker.assert_not_called()

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
    def test_batch_continues_after_bad_file_and_preserves_frame_metadata(self):
        """Return a per-item error for unreadable files without losing later inputs.
        Carry video presentation time and original decode index through image preparation.
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

    def test_failed_preparation_retains_frame_and_memory_pixels_for_review_copies(self):
        """Keep failed inputs' pixels and frame association available to later collection review.
        Do not lose the decoded frame or substitute the whole video file after a preparation error.
        """
        frame = Frame(self.image, self.root / "video.mp4", 12, 1.5)
        with patch.object(self.processor, "process_image", side_effect=ValueError("fixture")):
            failed_frame, failed_image = self.processor.process_images([frame, self.image])
        self.assertIs(failed_frame["original_image"], frame.image)
        self.assertEqual(failed_frame["frame_index"], 12)
        self.assertEqual(failed_frame["timestamp_seconds"], 1.5)
        self.assertIs(failed_image["original_image"], self.image)

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

    def test_colour_changes_cutout_not_retained_original_pixels(self):
        """Keep untouched RGB alongside the normalized head for the caller's colour measurements.
        Return correction provenance separately without performing any feature extraction.
        """
        self.processor.normalize_colour = Mock(return_value=(
            Image.new("RGB", self.image.size, "blue"), {"status": "fixture_correction"}))
        result = self.processor.process_image(self.image)
        self.assertEqual(result["image"].getpixel((80, 25))[:3], (0, 0, 255))
        self.assertEqual(result["colour"]["status"], "fixture_correction")
        self.assertEqual(result["original_image"].tobytes(), self.image.tobytes())

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
                self.processor.remove_background(self.image)
                raise RuntimeError("fixture")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.processor.remove_background(self.image)

    def test_default_cuda_uses_gpu_networks_without_loading_landmarker(self):
        """Default CUDA mode keeps both preparation CNNs on the selected GPU.
        Verify graph evidence using mocks without probing GPU hardware or creating a landmark task.
        """
        import cv2

        self.modnet.dump.return_value = 'layer [label="conv CUDA/CUDA"];'
        self.bisenet.dump.return_value = 'layer [label="conv CUDA/CUDA"];'
        processor = ImageProcessor(
            modnet_model=self.root / "modnet.model", bisenet_model=self.root / "bisenet.model")
        with processor, patch("ailias.imageprocessing.cuda.require_cuda") as require_cuda, patch(
            "cv2.cuda.setDevice"
        ) as select_device:
            result = processor.process_image(self.image, normalize_colour=False)
        self.assertEqual(require_cuda.call_count, 2)
        self.assertEqual(select_device.call_count, 2)
        self.modnet.setPreferableBackend.assert_called_once_with(cv2.dnn.DNN_BACKEND_CUDA)
        self.bisenet.setPreferableTarget.assert_called_once_with(cv2.dnn.DNN_TARGET_CUDA)
        self.create_landmarker.assert_not_called()
        self.assertEqual(result["execution"]["requested_device"], "cuda")
        self.assertEqual(result["execution"]["networks"]["modnet"]["status"], "verified")

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
        self.create_landmarker.assert_not_called()
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

    def test_separate_preparation_and_extraction_run_each_model_once(self):
        """Compose the public stages without rerunning matting or parsing for facial features.
        Closing the processor must not invalidate prepared pixels or the separately owned landmarker.
        """
        from ailias.imageprocessing.face_feature_extraction import FaceFeatureExtractor

        model = self.root / "mediapipe.model"
        model.write_bytes(b"unit fixture, not model weights")
        landmarker = Mock()
        points = [[.2 + i % 23 * .025, .1 + i // 23 * .014, 0] for i in range(478)]
        landmarker.detect.return_value = {
            "faces": [points], "blendshapes": [{}], "transforms": [np.eye(4).tolist()]}
        self.create_landmarker.side_effect = None
        self.create_landmarker.return_value = landmarker
        prepared = self.processor.process_image(self.image, normalize_colour=False)
        self.processor.close()
        with FaceFeatureExtractor(mediapipe_model=model, device="cpu") as extractor:
            result = extractor.extract_features(prepared)
        self.assertIsNotNone(result.features)
        self.modnet.forward.assert_called_once()
        self.bisenet.forward.assert_called_once()
        landmarker.detect.assert_called_once()
        landmarker.close.assert_called_once()

    def test_failed_video_preparation_reaches_grouping_review_without_losing_frame_identity(self):
        """Compose preparation, feature extraction and grouping with a failed frame and a usable successor.
        Preserve frame pixels/timing in review while reusing fake models and leaving originals untouched.
        """
        from ailias.imageprocessing.face_feature_extraction import FaceFeatureExtractor
        from ailias.imageprocessing.face_grouping import group_face_images

        model = self.root / "mediapipe.model"
        model.write_bytes(b"fixture, not model weights")
        landmarker = Mock()
        points = [[.2 + i % 23 * .025, .1 + i // 23 * .014, 0] for i in range(478)]
        landmarker.detect.return_value = {
            "faces": [points], "blendshapes": [{}], "transforms": [np.eye(4).tolist()]}
        self.create_landmarker.side_effect = None
        self.create_landmarker.return_value = landmarker
        self.modnet.forward.side_effect = [RuntimeError("fixture"),
                                          np.ones((1, 1, 512, 512), np.float32)]
        frame = Frame(self.image, self.root / "video.mp4", 12, 1.5)
        original_pixels = self.image.tobytes()
        with FaceFeatureExtractor(mediapipe_model=model, device="cpu") as extractor:
            prepared = self.processor.process_images([frame, self.image], normalize_colour=False)
            grouped = group_face_images(extractor.extract_features_batch(prepared),
                                        review_directory=self.root / "review")
        self.assertEqual(len(grouped.selected), 1)
        self.assertEqual(len(grouped.excluded), 1)
        excluded = grouped.excluded[0]
        self.assertEqual(excluded.extraction.frame_index, 12)
        self.assertEqual(excluded.extraction.timestamp_seconds, 1.5)
        with Image.open(excluded.review_copy) as copy:
            self.assertEqual(copy.tobytes(), original_pixels)
        self.assertEqual(self.image.tobytes(), original_pixels)
        landmarker.detect.assert_called_once()


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
