"""Sample reuse and fresh inference contracts, with no real models or hardware."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ailias.facefitting.evidence.photo_processing import (
    load_retained_frames,
    extract_photo_evidence,
)
from ailias.facefitting.configuration import VideoPolicy
from ailias.storage import Store, atomic_json


class SampledPhotosTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name), "subject")
        self.root = self.store.resolve("fresh")
        self.root.mkdir(parents=True)
        self.samples = []
        for index in range(5):
            image = self.root / f"retained-{index}.png"
            Image.new("RGB", (16, 12), (100, 80, 60)).save(image)
            self.samples.append({
                "face_id": f"sample-{index}", "source_id": "original-video",
                "image": self.store.reference_artifact(image), "pts": index*1000,
            })
        self.detector, self.parser = Mock(), Mock()
        self.parser.provenance = {"provider": "fake"}
        self.parser.parse.side_effect = lambda image, bounds: Image.new("L", image.size, 1)
        self.observed = {
            "source_xy": np.tile([8., 6.], (478, 1)).tolist(),
            "on_head": [True]*478, "face_box": [2, 2, 12, 10],
            "source_size": [16, 12], "eligible_identity": True, "reasons": [],
        }

    def test_only_images_and_capture_metadata_are_reused_not_predictions_or_source_video(self):
        manifest = self.root / "samples.json"
        atomic_json(manifest, {
            "status": "video_evidence_ready",
            "sources": {"missing-video": {"source": {"path": "nonexistent.mp4"}}},
            "frames": [{**frame, "source_xy": ["stale"], "blendshapes": {"stale": 1}}
                       for frame in self.samples],
        })
        samples, dependencies = load_retained_frames(self.store, manifest)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len(dependencies), 6)
        self.assertTrue(all("source_xy" not in frame and "blendshapes" not in frame for frame in samples))
        with patch("ailias.facefitting.evidence.photo_processing.analyze_face_frame", return_value=self.observed):
            result = extract_photo_evidence(self.store, samples, self.root, {}, VideoPolicy(),
                                     detector=self.detector, parser=self.parser)
        self.assertEqual(result["status"], "photo_evidence_ready")
        self.assertEqual(self.detector.detect.call_count, 5)
        self.assertEqual(self.parser.parse.call_count, 5)
        self.assertEqual(result["frames"][0]["source_xy"], self.observed["source_xy"])
        self.assertEqual(result["faces"]["sample-0"]["features"]["regions"]["skin"]["status"], "observed")
        self.assertTrue(all(self.store.verify(frame["image"]) for frame in samples))
        self.detector.close.assert_not_called()
        self.parser.close.assert_not_called()

    def test_corrupted_sample_fails_before_inference(self):
        manifest = self.root / "samples.json"
        atomic_json(manifest, {"status": "video_evidence_ready", "frames": self.samples})
        self.store.artifact_path(self.samples[0]["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            load_retained_frames(self.store, manifest)
        self.detector.detect.assert_not_called()

    def test_model_failure_keeps_failed_manifest_and_releases_both_owned_models(self):
        self.parser.parse.side_effect = RuntimeError("model failed")
        with patch("ailias.facefitting.evidence.photo_processing.DenseDetector", return_value=self.detector), \
                patch("ailias.facefitting.evidence.photo_processing.HeadParser", return_value=self.parser), \
                patch("ailias.facefitting.evidence.photo_processing.analyze_face_frame", return_value=self.observed):
            with self.assertRaisesRegex(RuntimeError, "sample-0") as caught:
                extract_photo_evidence(self.store, self.samples, self.root,
                                self.samples[0]["image"], VideoPolicy())
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        manifest = json.loads((self.root / "photo-evidence" / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.parser.close.assert_called_once()
        self.detector.close.assert_called_once()

    def test_parser_initialization_failure_still_releases_owned_detector(self):
        with patch("ailias.facefitting.evidence.photo_processing.DenseDetector", return_value=self.detector), \
                patch("ailias.facefitting.evidence.photo_processing.HeadParser", side_effect=ValueError("missing parser")):
            with self.assertRaisesRegex(ValueError, "missing parser"):
                extract_photo_evidence(self.store, self.samples, self.root,
                                self.samples[0]["image"], VideoPolicy())
        self.detector.close.assert_called_once()
        result = json.loads((self.root / "photo-evidence" / "manifest.json").read_text())
        self.assertEqual(result["status"], "failed")

    def test_cleanup_error_never_masks_primary_model_error(self):
        self.parser.parse.side_effect = RuntimeError("primary")
        self.detector.close.side_effect = OSError("secondary")
        with patch("ailias.facefitting.evidence.photo_processing.DenseDetector", return_value=self.detector), \
                patch("ailias.facefitting.evidence.photo_processing.HeadParser", return_value=self.parser), \
                patch("ailias.facefitting.evidence.photo_processing.analyze_face_frame", return_value=self.observed):
            with self.assertRaises(RuntimeError) as caught:
                extract_photo_evidence(self.store, self.samples, self.root,
                                self.samples[0]["image"], VideoPolicy())
        self.assertEqual(str(caught.exception.__cause__), "primary")
        result = json.loads((self.root / "photo-evidence" / "manifest.json").read_text())
        self.assertEqual(result["cleanup_errors"], ["OSError"])
