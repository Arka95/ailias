"""Use all accepted original-photo groups and preserve review exclusions and source pixels."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from ailias.facefitting.evidence.reviewed_photos import prepare_reviewed_photos
from ailias.storage import Store, atomic_json


class ReviewedPhotoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.store = Store(base, "subject")
        self.store.root.mkdir()
        self.root = self.store.resolve("new-fit")
        self.root.mkdir()
        sources, faces = {}, {}
        self.originals = []
        for index, decision in enumerate(("accept", "accept", "reject")):
            path = base / f"original-{index}.png"
            Image.new("RGB", (100, 80), (80+index, 40, 20)).save(path)
            reference = self.store.reference_artifact(path)
            face_id = f"face-{index}"
            sources[str(index)] = {"artifact": reference, "faces": [face_id],
                                   "metadata": {"oriented_size": [100, 80]}}
            faces[face_id] = {"id": face_id, "portrait_bounds": [10, 10, 90, 70],
                             "box_xywh": [20, 20, 40, 40],
                             "eligible": False, "group": f"group-{index}",
                             "curation": {"decision": decision}}
            self.originals.append((path, reference))
        atomic_json(self.store.resolve("imageprocessing\\runs\\reviewed\\manifest.json"), {
            "status": "complete", "sources": sources, "faces": faces})

    def test_all_accepted_groups_are_used_even_when_old_quality_flag_was_false(self):
        samples, dependencies = prepare_reviewed_photos(self.store, "reviewed", self.root)
        self.assertEqual(len(samples), 2)
        self.assertEqual({sample["source_id"] for sample in samples}, {"0", "1"})
        with Image.open(self.store.artifact_path(samples[0]["image"])) as image:
            self.assertEqual(image.size, (100, 80))
        self.assertEqual(samples[0]["image"], self.originals[0][1])
        self.assertEqual(samples[0]["detection_bounds"], [10, 10, 90, 70])
        self.assertTrue(all(self.store.verify(reference) for _, reference in self.originals))
        report = json.loads((self.root / "photo-inputs" / "manifest.json").read_text())
        self.assertEqual(report["source_count"], 3)
        self.assertEqual(report["excluded_faces"], [{"face_id": "face-2", "reason": "reject"}])
        self.assertFalse(report["previous_predictions_reused"])

    def test_changed_original_fails_with_a_persistent_specific_receipt(self):
        self.originals[0][0].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after review"):
            prepare_reviewed_photos(self.store, "reviewed", self.root)
        report = json.loads((self.root / "photo-inputs" / "manifest.json").read_text())
        self.assertEqual(report["status"], "failed")
