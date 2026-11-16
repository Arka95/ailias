import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from ailias.cli import main
from ailias.imageprocessing.cleanup import cleanup_alias
from ailias.imageprocessing.curation import (
    PROFILE, CurationError, capture_curation, load_filter,
)
from ailias.imageprocessing.pipeline import load_run, scan, select_group
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.imageprocessing.tests.test_processing import FakeBackend, face
from ailias.storage import Store


class CurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / "data", "arka")
        self.source = self.root / "photo.png"
        image = np.random.default_rng(42).integers(20, 230, (500, 800, 3), dtype=np.uint8)
        self.image = Image.fromarray(image)
        self.image.save(self.source)
        self.policy = ProcessingPolicy(min_face_size=80, min_laplacian_variance=1)
        self.faces = [face(100, [1, 0, 0]), face(500, [0, 1, 0])]
        self.seed = self.run_scan(self.faces)
        self.report = load_run(self.store, self.seed["run"])
        self.records = list(self.report["faces"].values())
        self.kept = self.store.resolve(self.records[0]["products"]["portrait"])
        self.rejected = self.store.resolve(self.records[1]["products"]["portrait"])
        self.rejected.unlink()

    def run_scan(self, faces, **options):
        return scan(self.store, self.source, self.policy, FakeBackend(faces), body=False, **options)

    def capture(self):
        return capture_curation(self.store, self.seed["run"], self.policy, True)

    def classifier(self):
        return load_filter(self.store, FakeBackend.provenance, self.policy)

    def test_capture_requires_explicit_review_and_never_restores_removed_portraits(self):
        with self.assertRaises(CurationError):
            capture_curation(self.store, self.seed["run"], self.policy)
        result = self.capture()
        self.assertEqual((result["kept"], result["rejected"]), (1, 1))
        self.assertFalse(self.rejected.exists())
        self.assertTrue(self.kept.exists())
        self.assertEqual(result["validation"]["counts"], {
            "kept": {"review": 1}, "rejected": {"review": 1},
        })

    def test_similarity_rejects_clear_negative_but_abstains_on_conflict_and_unknown(self):
        self.capture()
        classifier = self.classifier()
        clean = {"reasons": []}
        self.assertEqual(classifier.classify([1, 0.1, 0], clean)["decision"], "accept")
        self.assertEqual(classifier.classify([0.1, 1, 0], clean)["decision"], "reject")
        self.assertEqual(classifier.classify([1, 1, 0], clean)["decision"], "review")
        self.assertEqual(classifier.classify([0, 0, 1], clean)["decision"], "review")
        self.assertEqual(classifier.classify([1, 0.1, 0], {"reasons": ["blur"]})["decision"], "review")

    def test_repeat_pipeline_keeps_human_choices_and_does_not_publish_rejected_outputs(self):
        self.capture()
        result = self.run_scan(self.faces)
        report = load_run(self.store, result["run"])
        records = list(report["faces"].values())
        self.assertEqual(result["summary"]["portrait_filter"], {"accept": 1, "reject": 1})
        self.assertEqual(records[1]["products"], {})
        self.assertFalse(records[1]["eligible"])
        self.assertTrue(records[1]["identity_ambiguous"])
        self.assertEqual(records[1]["body"]["status"], "rejected_portrait")
        self.assertFalse(self.rejected.exists())
        self.assertTrue(self.kept.exists())
        self.assertEqual(len(report["groups"]), 1)
        selected = select_group(self.store, result["run"], report["groups"][0]["id"])
        self.assertEqual(selected["portraits"], [str(self.kept)])

    def test_identical_rejected_pixels_are_blocked_after_original_metadata_changes(self):
        self.capture()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("note", "Different original bytes; identical decoded pixels")
        self.image.save(self.source, pnginfo=metadata)
        result = self.run_scan(self.faces)
        self.assertEqual(result["summary"]["portrait_filter"], {"accept": 1, "reject": 1})
        self.assertFalse(self.rejected.exists())

    def test_unseen_negative_is_suppressed_and_uncertain_portraits_are_separate(self):
        self.capture()
        result = self.run_scan([face(300, [0.1, 1, 0])])
        report = load_run(self.store, result["run"])
        record = next(iter(report["faces"].values()))
        self.assertEqual(record["curation"]["reason"], "strong_rejected_exemplar_match")
        self.assertEqual(record["products"], {})
        self.assertEqual(report["groups"], [])
        result = self.run_scan([face(300, [0, 0, 1])])
        report = load_run(self.store, result["run"])
        record = next(iter(report["faces"].values()))
        self.assertEqual(record["curation"]["decision"], "review")
        portrait = self.store.resolve(record["products"]["portrait"])
        self.assertEqual(portrait.parent.name, "review")
        self.assertTrue(portrait.is_file())
        with self.assertRaisesRegex(ValueError, "no eligible"):
            select_group(self.store, result["run"], "face-001")

    def test_automatic_decisions_do_not_self_train_and_human_approval_updates_gallery(self):
        self.capture()
        result = self.run_scan([face(300, [0, 0, 1])])
        report = load_run(self.store, result["run"])
        path = self.store.resolve(next(iter(report["faces"].values()))["products"]["portrait"])
        update = capture_curation(self.store, result["run"], self.policy, True)
        self.assertEqual((update["kept"], update["rejected"], update["skipped_unreviewed"]), (1, 1, 1))
        approved = self.kept.parent / path.name
        path.replace(approved)
        update = capture_curation(self.store, result["run"], self.policy, True)
        self.assertEqual((update["kept"], update["rejected"]), (2, 1))
        result = self.run_scan([face(300, [0, 0, 1])])
        self.assertEqual(result["summary"]["portrait_filter"], {"accept": 1})

    def test_missing_kept_reference_stops_scan_instead_of_resurrecting_or_disabling_filter(self):
        self.capture()
        self.kept.unlink()
        with self.assertRaisesRegex(CurationError, "missing/changed"):
            self.run_scan(self.faces)
        self.assertFalse(self.kept.exists())

    def test_manual_deletion_during_inference_is_not_restored(self):
        self.capture()
        backend = FakeBackend(self.faces)

        def detect(image):
            self.kept.unlink()
            return self.faces

        backend.detect = detect
        with self.assertRaises(CurationError):
            scan(self.store, self.source, self.policy, backend, body=False)
        self.assertFalse(self.kept.exists())

    def test_changed_reference_and_changed_model_fail_closed(self):
        self.capture()
        with self.assertRaises(CurationError):
            load_filter(self.store, {"provider": "different-model"}, self.policy)
        self.kept.write_bytes(b"edited")
        with self.assertRaises(CurationError):
            self.classifier()
        with self.assertRaises(CurationError):
            self.capture()

    def test_embedding_dimension_mismatch_is_terminal_not_a_per_file_skip(self):
        self.capture()
        with self.assertRaises(CurationError):
            self.run_scan([face(300, [1, 0])])

    def test_quality_policy_cannot_delete_an_explicitly_kept_portrait(self):
        self.capture()
        strict = replace(self.policy, min_face_size=256)
        result = scan(self.store, self.source, strict, FakeBackend(self.faces), body=False)
        report = load_run(self.store, result["run"])
        record = next(iter(report["faces"].values()))
        self.assertEqual(record["curation"]["decision"], "accept")
        self.assertFalse(record["eligible"])
        self.assertTrue(self.kept.exists())

    def test_curation_protects_reference_assets_after_seed_run_is_pruned(self):
        self.capture()
        result = self.run_scan([self.faces[1]])
        cleanup_alias(self.store, result["run"], True)
        self.assertFalse(Path(self.seed["manifest"]).exists())
        self.assertTrue(self.kept.exists())
        self.assertIsNotNone(self.classifier())

    def test_no_kept_examples_cannot_enable_mass_rejection(self):
        self.kept.unlink()
        with self.assertRaises(CurationError):
            self.capture()
        self.assertFalse(self.store.resolve(PROFILE).exists())

    def test_cli_wires_review_confirmation_and_persists_profile(self):
        output, error = io.StringIO(), io.StringIO()
        args = ["--data-root", str(self.store.root.parent), "imageprocessing", "curate",
                "arka", self.seed["run"]]
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            self.assertEqual(main(args), 1)
            self.assertEqual(main([*args, "--removed-are-rejected"]), 0)
        profile = json.loads(self.store.resolve(PROFILE).read_text(encoding="utf-8"))
        self.assertEqual(len(profile["examples"]), 2)


if __name__ == "__main__":
    unittest.main()
