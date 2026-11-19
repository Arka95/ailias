"""Functional video-evidence units; no real media, model, GPU or subprocess."""

import copy
from fractions import Fraction
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.head_reconstruction.settings import VideoPolicy
from ailias.head_reconstruction.evidence.video_sampling import (
    extract_evidence,
    sampled_frames,
    select_identity,
    select_expression_aware,
    extract_audio,
)
from ailias.head_reconstruction.evidence.frame_analysis import analyze_face_frame
from ailias.storage import Store


def detection():
    points = np.random.default_rng(42).uniform([.3, .2, -.1], [.7, .8, .1], (478, 3))
    points[33, :2], points[263, :2] = [.35, .4], [.65, .4]
    return {"faces": [points.tolist()], "transforms": [np.eye(4).tolist()],
            "blendshapes": [{"jawOpen": .01, "eyeBlinkLeft": .01,
                            "eyeBlinkRight": .01, "mouthSmileLeft": .01}]}


class VideoEvidenceTests(unittest.TestCase):
    def test_processing_images_and_coordinates_share_resized_grid_without_changing_original(self):
        rgb = np.random.default_rng(42).integers(20, 230, (400, 800, 3), dtype=np.uint8)
        detected_sizes = []
        detector = SimpleNamespace(detect=lambda image: detected_sizes.append(image.shape) or detection())
        timing = {"pts": 0, "timestamp_seconds": 0, "decode_index": 0,
                  "time_base": "1/30", "container_time_seconds": 0, "video_start_seconds": 0}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input"
            source.mkdir()
            movie = source / "clip.mp4"
            movie.write_bytes(b"original")
            store = Store(root / "data", "person")
            with patch("ailias.head_reconstruction.evidence.video_sampling.sampled_frames", return_value=iter([(rgb, timing)])), \
                 patch("ailias.head_reconstruction.evidence.video_sampling.extract_audio", return_value={"status": "no_audio_stream"}):
                result = extract_evidence(store, source, VideoPolicy(
                    image_long_side=400, detection_long_side=256), detector=detector)
            frame = result["frames"][0]
            with Image.open(store.artifact_path(frame["image"])) as image:
                self.assertEqual(image.size, (400, 200))
            self.assertEqual(frame["source_size"], [400, 200])
            self.assertEqual(frame["original_source_size"], [800, 400])
            np.testing.assert_allclose(frame["source_xy"][33], [140, 80])
            self.assertEqual(detected_sizes, [(128, 256, 3)])
            self.assertEqual(movie.read_bytes(), b"original")

    def test_expression_and_blink_rejected_without_losing_source_coordinates(self):
        rgb = np.random.default_rng(2).integers(20, 230, (600, 1000, 3), dtype=np.uint8)
        value = detection()
        result = analyze_face_frame(rgb, value, VideoPolicy())
        self.assertTrue(result["eligible_identity"])
        np.testing.assert_allclose(result["source_xy"][33], [350, 240])
        self.assertAlmostEqual(result["eye_span_source"], 300)
        value["blendshapes"][0].update(jawOpen=.5, eyeBlinkLeft=.7)
        result = analyze_face_frame(rgb, value, VideoPolicy())
        self.assertFalse(result["eligible_identity"])
        self.assertIn("non_neutral_expression", result["reasons"])
        self.assertIn("blink", result["reasons"])

    def test_missing_face_or_pose_never_becomes_identity(self):
        rgb = np.zeros((600, 1000, 3), dtype=np.uint8)
        value = detection()
        value["faces"].append(value["faces"][0])
        self.assertFalse(analyze_face_frame(rgb, value, VideoPolicy())["eligible_identity"])
        value = detection()
        value["transforms"] = []
        self.assertIn("invalid_pose_transform", analyze_face_frame(rgb, value, VideoPolicy())["reasons"])

    def test_expressive_frames_require_expression_solver_and_neutral_gate_stays_closed(self):
        frames = [
            {"face_id": str(i), "source_id": "clip", "timestamp_seconds": float(i),
             "eligible_identity": False, "pose": {"category": "front"}, "quality_score": 50-i,
             "reasons": ["non_neutral_expression"], "quality": {"jaw_open": .15}}
            for i in range(8)]
        self.assertFalse(select_identity(frames, VideoPolicy())["training"])
        result = select_expression_aware(frames, VideoPolicy())
        self.assertTrue(result["training"])
        self.assertTrue(result["expression_model_required"])
        self.assertFalse(result["frames_are_neutral"])
        frames[0]["reasons"].append("blink")
        self.assertNotIn("0", select_expression_aware(frames, VideoPolicy())["training"])

    def test_audio_timing_gap_is_not_silently_reported_as_synchronized(self):
        def frame(pts):
            return SimpleNamespace(pts=pts, time_base=Fraction(1, 48000), samples=48,
                                   to_ndarray=lambda: np.zeros((1, 48), dtype=np.int16))
        container = SimpleNamespace(streams=SimpleNamespace(audio=[object()]),
                                    decode=lambda stream: iter([frame(0), frame(200)]))
        manager = unittest.mock.MagicMock()
        manager.__enter__.return_value = container
        av = SimpleNamespace(open=lambda path: manager,
                             AudioResampler=lambda **kw: SimpleNamespace(
                                 resample=lambda value: [] if value is None else [value]))
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"av": av}):
            with self.assertRaisesRegex(ValueError, "timestamp gap"):
                extract_audio(Path("fake.mp4"), Path(temp)/"audio.wav")

    def test_selection_balances_pose_and_enforces_temporal_validation_guard(self):
        frames = [
            {"face_id": str(i), "source_id": "clip", "timestamp_seconds": float(i),
             "eligible_identity": True, "pose": {"category": category}, "quality_score": 50-i}
            for i, category in enumerate(["front"]*8+["positive_oblique"]*4)
        ]
        original = copy.deepcopy(frames)
        result = select_identity(frames, VideoPolicy(max_identity_frames=10))
        by_id = {f["face_id"]: f for f in frames}
        self.assertEqual(len(result["pose_categories"]), 2)
        self.assertTrue(result["training"])
        self.assertTrue(result["validation"])
        for train in result["training"]:
            for validation in result["validation"]:
                self.assertGreaterEqual(abs(by_id[train]["timestamp_seconds"] -
                                            by_id[validation]["timestamp_seconds"]), 2)
        self.assertEqual(frames, original)

    def test_sampler_uses_variable_rate_pts_and_keeps_clock_origin(self):
        frames = [SimpleNamespace(pts=t, time_base=Fraction(1, 1000), rotation=0,
                                  to_ndarray=lambda **kw: np.zeros((2, 2, 3), dtype=np.uint8))
                  for t in (5000, 5020, 5500, 5510, 6300)]
        container = SimpleNamespace(
            streams=SimpleNamespace(video=[SimpleNamespace(metadata={}, frames=5)]), decode=lambda stream: iter(frames))
        manager = unittest.mock.MagicMock()
        manager.__enter__.return_value = container
        with patch.dict("sys.modules", {"av": SimpleNamespace(open=lambda path: manager)}):
            samples = list(sampled_frames(Path("fake.mp4"), VideoPolicy(frame_mode="sampled", sample_fps=2)))
        self.assertEqual([s[1]["pts"] for s in samples], [5000, 5500, 6300])
        self.assertAlmostEqual(samples[-1][1]["timestamp_seconds"], 1.3)
        self.assertEqual(samples[0][1]["video_start_seconds"], 5.)
        with patch.dict("sys.modules", {"av": SimpleNamespace(open=lambda path: manager)}):
            default = list(sampled_frames(Path("fake.mp4"), VideoPolicy()))
        self.assertEqual([s[1]["pts"] for s in default], [5000, 6300])
        with patch.dict("sys.modules", {"av": SimpleNamespace(open=lambda path: manager)}):
            full = list(sampled_frames(Path("fake.mp4"), VideoPolicy(frame_mode="all")))
        self.assertEqual([s[1]["pts"] for s in full], [5000, 5020, 5500, 5510, 6300])
        with patch.dict("sys.modules", {"av": SimpleNamespace(open=lambda path: manager)}):
            with self.assertRaisesRegex(ValueError, "decoding budget"):
                list(sampled_frames(Path("fake.mp4"), VideoPolicy(frame_mode="all", max_samples_per_video=2)))

    def test_failed_decode_is_recorded_and_original_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input"
            source.mkdir()
            movie = source / "clip.mp4"
            movie.write_bytes(b"fixture")
            store = Store(root / "data", "person")
            detector = SimpleNamespace()
            with patch("ailias.head_reconstruction.evidence.video_sampling.sampled_frames", side_effect=ValueError("decode failure")):
                with self.assertRaises(ValueError):
                    extract_evidence(store, source, VideoPolicy(), detector=detector)
            import json
            manifest = next((store.root / "facefitting" / "video").glob("*/manifest.json"))
            self.assertEqual(json.loads(manifest.read_text())["status"], "failed")
            self.assertEqual(movie.read_bytes(), b"fixture")

    def test_policy_rejects_nonfinite_or_unbounded_work(self):
        for value in (0, 9000, 1920.5, True):
            with self.assertRaises(ValueError):
                VideoPolicy(image_long_side=value)
        for value in ([], {}, None, "ALL"):
            with self.assertRaises(ValueError):
                VideoPolicy(frame_mode=value)
        for value in (float("nan"), 0, 30, True):
            with self.assertRaises(ValueError):
                VideoPolicy(sample_fps=value)
