"""Verify the structured solver boundary using numerical arrays and fake runtime metadata."""

import json
import unittest

import numpy as np

from ailias.head_reconstruction.geometry.fitting_results import (
    fitting_result_from_solver,
)


class FittingResultTests(unittest.TestCase):
    def solver_result(self):
        """Build a backend record with a known view association and synthetic CUDA provenance.
        Supply data only: this fixture neither imports Torch nor executes a GPU/model.
        """
        return {
            "identity": np.array([0.25]),
            "per_view": np.zeros((1, 8)),
            "projected": np.zeros((1, 468, 2)),
            "parameters": np.zeros(9),
            "report": {
                "identity_modes": 1,
                "expression_modes": 1,
                "median_outer_eye_span_error": 0.02,
                "backend": "synthetic-cuda",
                "device": "cuda:7",
                "seconds": 0.1,
                "nfev": 3,
                "converged": True,
                "termination": "fixture",
                "objective": 0.001,
                "gpu_name": "fixture, not probed",
                "jacobian_evaluations": 2,
                "views": [
                    {
                        "face_id": "photo-A",
                        "supported_points": 180,
                        "median_outer_eye_span_error": 0.02,
                        "p90_outer_eye_span_error": 0.03,
                        "weighted_rms_outer_eye_span_error": 0.025,
                        "median_source_pixel_error": 2.5,
                        "focal_factor": 1.2,
                        "view": {
                            "yaw_deg": 10.0,
                            "pitch_deg": 2.0,
                            "category": "front",
                            "calibrated": False,
                        },
                    }
                ],
            },
        }

    def test_typed_result_preserves_arrays_view_association_and_backend_specific_metadata(
        self,
    ):
        """Keep hardware provenance and numerical precision when adapting an existing solver result.
        Serialize nested typed records/arrays into strict JSON without exposing legacy face-ID field names.
        """
        original = self.solver_result()
        result = fitting_result_from_solver(original)
        np.testing.assert_array_equal(
            result.parameters.identity_coefficients, original["identity"]
        )
        self.assertEqual(result.summary.views[0].view_id, "photo-A")
        self.assertEqual(result.summary.runtime.evaluations, 3)
        self.assertEqual(
            result.summary.runtime.backend_details,
            {
                "gpu_name": "fixture, not probed",
                "jacobian_evaluations": 2,
            },
        )
        data = json.loads(str(result))
        self.assertEqual(data["summary"]["views"][0]["view_id"], "photo-A")
        self.assertNotIn("face_id", data["summary"]["views"][0])
        self.assertEqual(data["parameters"]["identity_coefficients"], [0.25])

    def test_nested_nonfinite_numerical_arrays_cannot_be_exported_as_json(self):
        """Reject invalid numerical values after nested parameter-array conversion.
        Do not produce a successful-looking snapshot containing nonstandard NaN tokens.
        """
        result = fitting_result_from_solver(self.solver_result())
        result.parameters.optimizer_parameters[0] = np.nan
        with self.assertRaises(ValueError):
            result.to_dict()
