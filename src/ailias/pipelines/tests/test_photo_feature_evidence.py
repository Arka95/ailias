"""Fresh photo evidence, crop grids and resource ownership without neural inference."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ailias.imageprocessing.face_features import FaceExtractionResult
from ailias.imageprocessing.images import ImageProcessor
from ailias.pipelines.photo_feature_evidence import extract_photo_feature_evidence
from ailias.pipelines.photo_inputs import load_sampled_photo_inputs
from ailias.pipelines.tests.photo_fixtures import make_photo_features
from ailias.storage import Store, atomic_json


MODULE = "ailias.pipelines.photo_feature_evidence."


class PhotoFeatureEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name), "subject")
        self.root = self.store.resolve("fresh")
        self.root.mkdir(parents=True)
        self.samples = []
        for index in range(5):
            image = self.root / f"retained-{index}.png"
            with Image.new("RGB", (160, 128), (100, 80, 60)) as pixels:
                pixels.save(image)
            self.samples.append(
                {
                    "face_id": f"sample-{index}",
                    "source_id": "original-video",
                    "image": self.store.reference_artifact(image),
                    "pts": index * 1000,
                }
            )
        self.processor, self.extractor = Mock(spec=ImageProcessor), Mock()
        self.processor.load_image = ImageProcessor.load_image
        self.processor.crop_image = ImageProcessor.crop_image
        self.processor.process_image.side_effect = lambda image: {
            "image": image.convert("RGBA"),
            "original_image": image.copy(),
            "regions": Image.new("L", image.size, 1),
        }
        self.extractor.extract_feature_result.side_effect = self.feature_result

    def feature_result(self, prepared):
        return FaceExtractionResult(
            features=make_photo_features(
                prepared["image"].size,
                warnings=("blurred", "non_neutral_expression"),
                yaw=75,
                pitch=35,
            ),
            image=prepared["original_image"],
            head_image=prepared["image"],
            source=prepared["source"],
            frame_index=prepared["frame_index"],
        )

    def extract(self, samples=None, *, owned=False):
        kwargs = (
            {} if owned else {"processor": self.processor, "extractor": self.extractor}
        )
        return extract_photo_feature_evidence(
            self.store,
            self.samples if samples is None else samples,
            self.root,
            self.samples[0]["image"],
            device="cpu",
            **kwargs,
        )

    def manifest(self):
        return json.loads((self.root / "photo-evidence" / "manifest.json").read_text())

    def test_only_pixels_and_capture_metadata_are_reused_and_all_usable_angles_retained(
        self,
    ):
        manifest = self.root / "samples.json"
        atomic_json(
            manifest,
            {
                "status": "video_evidence_ready",
                "sources": {"missing-video": {"source": {"path": "nonexistent.mp4"}}},
                "frames": [
                    {**sample, "source_xy": ["stale"], "blendshapes": {"stale": 1}}
                    for sample in self.samples
                ],
            },
        )
        samples, dependencies = load_sampled_photo_inputs(self.store, manifest)
        self.assertEqual(len(dependencies), 6)
        self.assertTrue(
            all(
                "source_xy" not in item and "blendshapes" not in item
                for item in samples
            )
        )
        batch = self.extract(iter(samples))
        self.assertTrue(all(frame.selected for frame in batch.frames))
        self.assertTrue(
            all(frame.view_group == "right_profile_up" for frame in batch.frames)
        )
        self.assertEqual(self.processor.process_image.call_count, 5)
        self.assertEqual(self.extractor.extract_feature_result.call_count, 5)
        self.assertTrue(all(self.store.verify(frame.image) for frame in batch.frames))
        self.processor.close.assert_not_called()
        self.extractor.close.assert_not_called()
        saved = self.manifest()
        self.assertEqual(saved["status"], "photo_evidence_ready")
        self.assertEqual(saved["frames"][0]["features"]["source_size"], [160, 128])
        self.assertTrue(all(self.store.verify(ref) for ref in batch.artifacts.values()))

    def test_reviewed_crop_is_saved_as_the_actual_grid_and_original_provenance_is_retained(
        self,
    ):
        sample = {
            **self.samples[0],
            "reviewed_crop": [20, 10, 140, 110],
            "original_photo": self.samples[0]["image"],
            "original_source_size": [160, 128],
        }
        frame = self.extract([sample]).frames[0]
        self.assertEqual(frame.features.source_size, (120, 100))
        self.assertEqual(frame.capture["original_photo"], sample["image"])
        self.assertNotEqual(frame.image["path"], sample["image"]["path"])
        for reference in (frame.image, frame.regions, frame.head_image):
            with Image.open(self.store.artifact_path(reference)) as image:
                self.assertEqual(image.size, (120, 100))
        self.assertTrue(self.store.verify(sample["image"]))

    def test_ambiguous_face_is_excluded_with_review_copy_not_substituted(self):
        def ambiguous(prepared):
            return FaceExtractionResult(
                image=prepared["original_image"],
                source=prepared["source"],
                reason="multiple_faces",
                error_type="ValueError",
            )

        self.extractor.extract_feature_result.side_effect = ambiguous
        frame = self.extract(self.samples[:1]).frames[0]
        self.assertFalse(frame.selected)
        self.assertEqual(frame.reason, "multiple_faces")
        self.assertTrue(self.store.verify(frame.review_copy))
        self.assertTrue(self.store.verify(self.samples[0]["image"]))

    def test_corrupted_sample_fails_before_preparation(self):
        manifest = self.root / "samples.json"
        atomic_json(
            manifest, {"status": "video_evidence_ready", "frames": self.samples}
        )
        self.store.artifact_path(self.samples[0]["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            load_sampled_photo_inputs(self.store, manifest)
        self.processor.process_image.assert_not_called()

    def test_owned_resources_close_without_masking_primary_extraction_failure(self):
        self.extractor.extract_feature_result.side_effect = RuntimeError("primary")
        self.processor.close.side_effect = OSError("secondary")
        with (
            patch(
                MODULE + "create_photo_image_processor",
                return_value=(self.processor, {}),
            ),
            patch(MODULE + "FaceFeatureExtractor", return_value=self.extractor),
        ):
            with self.assertRaisesRegex(RuntimeError, "sample-0") as caught:
                self.extract(owned=True)
        self.assertEqual(str(caught.exception.__cause__), "primary")
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(self.manifest()["cleanup_errors"], ["OSError"])
        self.extractor.close.assert_called_once()
        self.processor.close.assert_called_once()

    def test_extractor_initialization_failure_still_closes_owned_preparation(self):
        with (
            patch(
                MODULE + "create_photo_image_processor",
                return_value=(self.processor, {}),
            ),
            patch(
                MODULE + "FaceFeatureExtractor", side_effect=ValueError("missing model")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "missing model"):
                self.extract(owned=True)
        self.processor.close.assert_called_once()
        self.assertEqual(self.manifest()["status"], "failed")

    def test_source_change_during_extraction_fails_persistently(self):
        def corrupt(prepared):
            result = self.feature_result(prepared)
            self.store.artifact_path(self.samples[0]["image"]).write_bytes(b"changed")
            return result

        self.extractor.extract_feature_result.side_effect = corrupt
        with self.assertRaisesRegex(ValueError, "changed during extraction"):
            self.extract(self.samples[:1])
        self.assertEqual(self.manifest()["status"], "failed")
