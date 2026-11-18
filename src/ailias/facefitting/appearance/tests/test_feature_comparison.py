import copy
import json
import tempfile
import unittest
from pathlib import Path

from ailias.facefitting.appearance.feature_comparison import compare_features
from ailias.facefitting.unreal.character_features_worker import validate_profile
from ailias.storage import Store, atomic_json


class FeatureComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.store = Store(root / "data", "test")
        self.reference_path, self.candidate_path = root / "photo.json", root / "render.json"
        self.source = root / "source.png"
        self.source.write_bytes(b"synthetic image dependency")
        self.reference = {
            "schema_version": 1, "method": "photo-regions-and-projected-landmarks-v1",
            "status": "features_extracted_awaiting_review", "cohort": "frontal photo",
            "input_dependencies": [self.store.reference_artifact(self.source)], "artifacts": {},
            "faces": {"face": {"status": "observed", "expression": {"mouthSmileLeft": .9}, "features": {
                "geometry_2d": {"nose_width": {"status": "observed", "outer_eye_span_ratio": .4}},
                "iris": {"status": "observed", "median_linear_rgb": [.1, .05, .02]},
            }}},
        }
        self.candidate = copy.deepcopy(self.reference)
        self.candidate.update(cohort="native render", image_role="render")
        self.candidate["faces"]["face"]["features"]["geometry_2d"]["nose_width"]["outer_eye_span_ratio"] = .3
        self.candidate["faces"]["face"]["features"]["iris"]["median_linear_rgb"] = [.2, .1, .1]
        self.save_inputs()

    def save_inputs(self):
        atomic_json(self.reference_path, self.reference)
        atomic_json(self.candidate_path, self.candidate)

    def compare(self, tint=None):
        return compare_features(self.store, self.reference_path, "face", self.candidate_path, tint)

    def test_comparison_wires_photo_evidence_to_a_validated_but_unapplied_native_proposal(self):
        report = self.compare([1, 1, 1])
        self.assertAlmostEqual(report["geometry_2d"]["nose_width"]["signed_delta"], -.1)
        path = self.store.artifact_path(report["artifacts"]["iris_proposal"])
        profile = validate_profile(json.loads(path.read_text()))
        for actual, expected in zip(profile["eyes"]["global_tint_linear"], [.5, .5, .2]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(self.store.verify(item) for item in profile["sources"]))
        self.assertTrue(report["iris_proposal"]["render_verification_required"])
        self.assertFalse(report["native_edits_applied"])
        self.assertFalse(report["likeness_verified"])
        self.assertEqual(json.loads(Path(report["manifest"]).read_text())["revision"], report["revision"])

    def test_insufficient_iris_evidence_preserves_geometry_without_inventing_a_profile(self):
        self.candidate["faces"]["face"]["features"]["iris"] = {"status": "review"}
        self.save_inputs()
        report = self.compare([1, 1, 1])
        self.assertEqual(report["iris_proposal"]["status"], "review")
        self.assertNotIn("iris_proposal", report["artifacts"])
        self.assertIn("nose_width", report["geometry_2d"])

    def test_stale_dependency_or_unmarked_render_cannot_create_a_native_proposal(self):
        self.candidate["image_role"] = "reference"
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "rendered face"):
            self.compare([1, 1, 1])
        self.candidate["image_role"] = "render"
        self.save_inputs()
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.compare([1, 1, 1])

    def test_unobservable_geometry_is_reported_instead_of_assigned_zero_error(self):
        self.candidate["faces"]["face"]["features"]["geometry_2d"]["nose_width"]["status"] = "review"
        self.save_inputs()
        report = self.compare()
        self.assertEqual(report["geometry_2d"], {})
        self.assertEqual(report["unmeasurable_geometry"], ["nose_width"])
        self.assertNotIn("iris_proposal", report)
