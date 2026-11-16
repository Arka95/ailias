import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ailias.geometry.metrics import compare_geometry
from ailias.geometry.comparison import compare_files
from ailias.storage import Store


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.reference = {
            "dimension": 3, "units": "mm",
            "landmarks": {"left_eye": [-30, 0, 0], "right_eye": [30, 0, 0],
                          "nose": [0, -20, 20], "chin": [0, -65, -5],
                          "forehead": [0, 35, -8]},
            "regions": {"eyes": ["left_eye", "right_eye"], "nose": ["nose"]},
        }

    def transformed(self, scale=1):
        candidate = copy.deepcopy(self.reference)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for name, point in candidate["landmarks"].items():
            candidate["landmarks"][name] = (scale * np.array(point) @ rotation + [10, 20, 30]).tolist()
        return candidate

    def test_rigid_alignment_removes_pose_but_not_shape_difference(self):
        result = compare_geometry(self.reference, self.transformed())
        self.assertLess(result["landmark_error"]["rmse"], 1e-10)
        changed = self.transformed()
        changed["landmarks"]["nose"][2] += 10
        result = compare_geometry(self.reference, changed)
        self.assertGreater(result["normalized_rmse"], 0)
        self.assertGreater(result["regions"]["nose"]["mean"], 0)

    def test_metric_vs_scale_invariant_fits_are_explicit(self):
        rigid = compare_geometry(self.reference, self.transformed(2))
        relative = compare_geometry(self.reference, self.transformed(2), "similarity")
        self.assertGreater(rigid["landmark_error"]["rmse"], 1)
        self.assertLess(relative["landmark_error"]["rmse"], 1e-10)
        self.assertAlmostEqual(relative["candidate_to_reference"]["scale"], 0.5)

    def test_reflection_is_not_accepted_as_rigid_matching_face(self):
        mirrored = copy.deepcopy(self.reference)
        for point in mirrored["landmarks"].values():
            point[0] *= -1
        result = compare_geometry(self.reference, mirrored)
        self.assertGreater(result["landmark_error"]["rmse"], 0.1)
        self.assertAlmostEqual(np.linalg.det(result["candidate_to_reference"]["rotation"]), 1)

    def test_3d_units_are_converted_to_millimeters(self):
        meters = copy.deepcopy(self.reference)
        meters["units"] = "m"
        meters["landmarks"] = {name: [v / 1000 for v in point]
                               for name, point in meters["landmarks"].items()}
        result = compare_geometry(self.reference, meters)
        self.assertEqual(result["units"], "mm")
        self.assertLess(result["landmark_error"]["rmse"], 1e-10)

    def test_sampled_surface_errors_have_known_euclidean_units(self):
        reference, candidate = copy.deepcopy(self.reference), copy.deepcopy(self.reference)
        reference["surface_points"] = [[0, 0, 0]]
        candidate["surface_points"] = [[3, 4, 0]]
        result = compare_geometry(reference, candidate)["sampled_surface"]
        self.assertAlmostEqual(result["symmetric_chamfer_mean"], 5)
        self.assertAlmostEqual(result["hausdorff_max"], 5)
        self.assertAlmostEqual(result["hausdorff_p95"], 5)

    def test_2d_comparison_is_labelled_projected_not_3d(self):
        reference = {"dimension": 2, "units": "pixels",
                     "landmarks": {"left_eye": [0, 0], "right_eye": [20, 0], "nose": [10, 20]}}
        result = compare_geometry(reference, reference, "similarity")
        self.assertFalse(result["is_3d_geometry"])
        self.assertLess(result["normalized_rmse"], 1e-10)

    def test_missing_correspondence_and_nonfinite_values_are_rejected(self):
        missing = copy.deepcopy(self.reference)
        missing["landmarks"].pop("nose")
        with self.assertRaisesRegex(ValueError, "identical named"):
            compare_geometry(self.reference, missing)
        invalid = copy.deepcopy(self.reference)
        invalid["landmarks"]["nose"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_geometry(self.reference, invalid)

    def test_degenerate_points_and_unbounded_surface_inputs_fail(self):
        invalid = copy.deepcopy(self.reference)
        invalid["landmarks"] = {name: [0, 0, 0] for name in invalid["landmarks"]}
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            compare_geometry(invalid, invalid)
        large = copy.deepcopy(self.reference)
        large["surface_points"] = [[0, 0, 0]] * 10001
        with self.assertRaisesRegex(ValueError, "10000"):
            compare_geometry(large, large)

    def test_file_comparison_keeps_verified_input_snapshots_with_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "reference.json", root / "candidate.json"
            first.write_text(json.dumps(self.reference), encoding="utf-8")
            second.write_text(json.dumps(self.transformed()), encoding="utf-8")
            store = Store(root / "data", "synthetic")
            result = compare_files(store, first, second, "rigid")
            self.assertTrue(Path(result["report"]).is_file())
            self.assertLess(result["landmark_error"]["rmse"], 1e-10)
            first.write_text("changed after comparison", encoding="utf-8")
            self.assertTrue(all(store.verify(artifact) for artifact in result["inputs"].values()))

    def test_invalid_comparison_cleans_only_its_uncommitted_input_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "reference.json", root / "candidate.json"
            first.write_text("invalid json", encoding="utf-8")
            second.write_text(json.dumps(self.reference), encoding="utf-8")
            store = Store(root / "data", "synthetic")
            earlier = store.resolve(str(Path("geometry") / "previous" / "report.json"))
            earlier.parent.mkdir(parents=True)
            earlier.write_text("previous result", encoding="utf-8")
            with self.assertRaises(ValueError):
                compare_files(store, first, second, "rigid")
            self.assertEqual(list(store.resolve("geometry").rglob("*.*")), [earlier])
