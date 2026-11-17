import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.cli import execute, parser
from ailias.imageprocessing.feature_evidence import extract_features, inspect_image
from ailias.imageprocessing.tests.test_appearance_features import eye_image
from ailias.storage import Store, atomic_json


class FakeParser:
    provenance = {"provider": "synthetic-parts", "device": "cpu"}

    def __init__(self):
        self.calls = 0
        self.closed = False

    def parse(self, image, bounds):
        self.calls += 1
        return Image.new("L", image.size, 1)

    def close(self):
        self.closed = True


class FeatureEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / "data", "test")
        self.store.root.mkdir(parents=True)
        rgb, points = eye_image()
        source = self.root / "original.png"
        Image.fromarray(rgb).save(source)
        portrait = self.store.root / "portrait.png"
        Image.fromarray(rgb).save(portrait)
        aligned = self.store.root / "aligned.png"
        Image.fromarray(rgb).convert("RGBA").save(aligned)
        curation = self.store.resolve(r"imageprocessing\curation.json")
        atomic_json(curation, {"synthetic": True})
        transform = np.eye(3).tolist()
        analysis = {
            "aligned_cutout": self.store.reference_artifact(aligned),
            "alignment": {"status": "ready", "source_to_aligned": transform,
                          "aligned_to_source": transform},
        }
        source_ref = self.store.reference_artifact(source)
        entry = {
            "face_id": "photo-face", "source_id": source_ref["sha256"], "source": source_ref,
            "portrait": self.store.reference_artifact(portrait),
            "portrait_bounds": [0, 0, 128, 64], "box_xywh": [0, 0, 128, 64],
            "analysis": analysis, "view": {"category": "front"},
        }
        self.prepared = {"inputs": {"schema_version": 1,
                                   "curation_sha256": self.store.reference_artifact(curation)["sha256"],
                                   "entries": [entry]}}
        self.dense = {"method": "mediapipe-478-reversible-source-v1", "faces": [{
            "face_id": entry["face_id"], "source_id": entry["source_id"], "status": "ready",
            **copy.deepcopy(analysis), "source_size": [128, 64],
            "source_xy": points.tolist(), "aligned_xy": points.tolist(), "on_head": [True]*478,
            "blendshapes": {"jawOpen": .1},
        }]}
        self.inputs_path, self.observations_path = self.root / "sources.json", self.root / "observations.json"
        self.backend = FakeParser()
        self.save_inputs()

    def save_inputs(self):
        atomic_json(self.inputs_path, self.prepared)
        atomic_json(self.observations_path, self.dense)

    def run_features(self, **kwargs):
        return extract_features(self.store, self.inputs_path, self.observations_path,
                                "same appearance", self.backend, **kwargs)

    def test_cli_writes_reviewable_evidence_without_dense_inference_or_native_mutation(self):
        args = parser().parse_args([
            "--data-root", str(self.root / "data"), "imageprocessing", "features", "test",
            "--inputs", str(self.inputs_path), "--observations", str(self.observations_path),
            "--cohort", "same appearance",
        ])
        original = (self.root / "original.png").read_bytes()
        with patch("ailias.imageprocessing.faceparsing.HeadParser", return_value=self.backend):
            result, code = execute(args)
        self.assertEqual(code, 0)
        self.assertTrue(self.backend.closed)
        persisted = json.loads(Path(result["manifest"]).read_text())
        record = persisted["faces"]["photo-face"]
        self.assertEqual(record["features"]["iris"]["status"], "observed")
        self.assertEqual(record["expression"], {"jawOpen": .1})
        self.assertEqual(record["cohort"], "same appearance")
        self.assertTrue(self.store.verify(record["regions"]))
        self.assertTrue(all(self.store.verify(item) for item in persisted["input_dependencies"]))
        self.assertFalse(persisted["native_edits_applied"])
        self.assertEqual(persisted["observation_mode"], "explicit_reuse_no_dense_inference")
        self.assertEqual((self.root / "original.png").read_bytes(), original)
        self.assertFalse(self.store.resolve(".lock").exists())

    def test_stale_curation_or_changed_source_stops_before_parser(self):
        for scenario in ("curation", "source"):
            with self.subTest(scenario=scenario):
                if scenario == "curation":
                    self.prepared["inputs"]["curation_sha256"] = "0"*64
                else:
                    self.prepared["inputs"]["curation_sha256"] = self.store.reference_artifact(
                        self.store.resolve(r"imageprocessing\curation.json"))["sha256"]
                    (self.root / "original.png").write_bytes(b"changed")
                self.save_inputs()
                with self.assertRaisesRegex(ValueError, "curation|evidence changed"):
                    self.run_features()
        self.assertEqual(self.backend.calls, 0)

    def test_mismatched_source_or_transform_cannot_reuse_dense_coordinates(self):
        for scenario in ("source", "transform", "points", "duplicate"):
            original = copy.deepcopy(self.dense)
            with self.subTest(scenario=scenario):
                if scenario == "source":
                    self.dense["faces"][0]["source_id"] = "another-photo"
                elif scenario == "transform":
                    self.dense["faces"][0]["alignment"]["aligned_to_source"][0][2] = 5
                elif scenario == "points":
                    self.dense["faces"][0]["source_xy"][0][0] += 10
                else:
                    self.dense["faces"].append(copy.deepcopy(self.dense["faces"][0]))
                self.save_inputs()
                with self.assertRaisesRegex(ValueError, "source|coordinates|round-trip|unique"):
                    self.run_features()
            self.dense = original
        self.assertEqual(self.backend.calls, 0)

    def test_review_observation_is_retained_and_unknown_face_is_not_silently_skipped(self):
        self.dense["faces"][0].update(status="review", reason="occluded")
        self.save_inputs()
        report = self.run_features()
        self.assertEqual(report["faces"]["photo-face"]["reason"], "occluded")
        self.assertEqual(report["summary"], {"review": 1})
        self.assertEqual(self.backend.calls, 0)
        with self.assertRaisesRegex(ValueError, "Selected face"):
            self.run_features(face_ids=["missing"])

    def test_mid_run_source_change_marks_the_retained_job_failed(self):
        original_parse = self.backend.parse

        def mutate(image, bounds):
            result = original_parse(image, bounds)
            (self.root / "original.png").write_bytes(b"changed")
            return result

        self.backend.parse = mutate
        with self.assertRaisesRegex(ValueError, "during extraction"):
            self.run_features()
        manifests = list(self.store.resolve(r"imageprocessing\features").glob("*\\manifest.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(manifests[0].read_text())["status"], "failed")

    def test_explicit_render_inspection_uses_source_pixels_and_preserves_expression(self):
        _, points = eye_image()
        detection = {
            "faces": [np.c_[points/[128, 64], np.zeros(478)].tolist()],
            "blendshapes": [{"mouthSmileLeft": .05}], "transforms": [],
        }
        model = self.root / "landmarker.task"
        model.write_bytes(b"synthetic model")
        from types import SimpleNamespace
        detector = SimpleNamespace(detect=lambda rgb: detection)
        report = inspect_image(self.store, self.root / "original.png", "native draft",
                               self.backend, detector, self.store.reference_artifact(model), "render")
        face = next(iter(report["faces"].values()))
        np.testing.assert_allclose(face["source_xy"], points)
        self.assertEqual(face["expression"], {"mouthSmileLeft": .05})
        self.assertEqual(face["features"]["iris"]["status"], "observed")
        self.assertEqual(report["image_role"], "render")

    def test_render_inspection_abstains_for_missing_or_multiple_faces(self):
        from types import SimpleNamespace
        model = self.root / "landmarker.task"
        model.write_bytes(b"synthetic model")
        for faces in ([], [[], []]):
            with self.subTest(faces=len(faces)):
                detector = SimpleNamespace(detect=lambda rgb: {"faces": faces})
                report = inspect_image(self.store, self.root / "original.png", "native draft",
                                       self.backend, detector, self.store.reference_artifact(model), "render")
                self.assertEqual(report["status"], "review")
                self.assertEqual(report["faces"], {})
                self.assertTrue(Path(report["manifest"]).is_file())
        self.assertEqual(self.backend.calls, 0)
