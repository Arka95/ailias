"""Shared identity selection and original-view ordering, without running a fitter."""

import unittest
from unittest.mock import patch

import numpy as np

from ailias.head_reconstruction.geometry.identity_fitting import (
    fit_shared_head_identity,
)
from ailias.head_reconstruction.geometry.tests.fitting_fixtures import make_solver_fit
from ailias.head_reconstruction.reconstruction_data import (
    ReconstructionInputs,
    ReconstructionReport,
)
from ailias.head_reconstruction.settings import FitPolicy


class SharedIdentityTests(unittest.TestCase):
    def fixture(self):
        views = [{"face_id": str(index)} for index in range(3)]
        runs = [
            {"per_view": np.zeros((2, 8))},
            {"per_view": np.zeros((1, 8)), "report": {}},
            {
                "identity": np.array([0.25]),
                "per_view": np.array([np.arange(10.0, 18.0), np.arange(30.0, 38.0)]),
            },
            {"per_view": np.array([np.arange(20.0, 28.0)]), "report": {}},
            make_solver_fit(["0", "1", "2"], identity=np.array([0.25])),
        ]
        score = lambda value: {
            "view_balanced_mean_error": value,
            "per_initial_view_category": {"front": value},
        }
        return views, runs, score

    def test_final_fit_restores_training_and_heldout_cameras_to_original_view_order(
        self,
    ):
        views, runs, score = self.fixture()
        policy = FitPolicy(identity_candidates=(1,))
        report = ReconstructionReport(policy, ReconstructionInputs(len(views)))
        with (
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.validation_indices",
                return_value=[1],
            ),
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.fit",
                side_effect=runs,
            ) as fit,
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.evaluation_score",
                side_effect=[score(1), score(0.8)],
            ),
        ):
            result = fit_shared_head_identity(object(), views, policy, report)
        np.testing.assert_array_equal(result.parameters.identity_coefficients, [0.25])
        np.testing.assert_array_equal(
            fit.call_args.kwargs["initial"],
            np.r_[
                0.25,
                np.arange(10.0, 18.0),
                np.arange(20.0, 28.0),
                np.arange(30.0, 38.0),
            ],
        )
        self.assertEqual(report.inputs.training_view_ids, ["0", "2"])
        self.assertEqual(report.inputs.validation_view_ids, ["1"])

    def test_insufficient_heldout_improvement_does_not_run_a_final_personalized_fit(
        self,
    ):
        views, runs, score = self.fixture()
        policy = FitPolicy(identity_candidates=(1,))
        report = ReconstructionReport(policy, ReconstructionInputs(len(views)))
        with (
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.validation_indices",
                return_value=[1],
            ),
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.fit",
                side_effect=runs,
            ) as fit,
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.evaluation_score",
                side_effect=[score(1), score(0.995)],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "criterion failed"):
                fit_shared_head_identity(object(), views, policy, report)
        self.assertEqual(fit.call_count, 4)

    def test_completed_candidate_diagnostics_survive_a_later_solver_failure(self):
        """Retain a successful candidate evaluation if a subsequent capacity cannot finish fitting.
        Leave failure classification to the orchestration layer without losing completed selection evidence.
        """
        views, runs, score = self.fixture()
        policy = FitPolicy(identity_candidates=(1, 2))
        report = ReconstructionReport(policy, ReconstructionInputs(len(views)))
        with (
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.validation_indices",
                return_value=[1],
            ),
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.fit",
                side_effect=runs[:4] + [RuntimeError("fixture later solver failure")],
            ),
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.evaluation_score",
                side_effect=[score(1), score(0.8)],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "later solver"):
                fit_shared_head_identity(object(), views, policy, report)
        self.assertEqual(report.inputs.training_view_ids, ["0", "2"])
        self.assertEqual(report.inputs.validation_view_ids, ["1"])
        self.assertEqual(len(report.candidates), 1)
        self.assertEqual(report.candidates[0].identity_modes, 1)
        self.assertAlmostEqual(report.candidates[0].score.view_balanced_mean_error, 0.8)

    def test_lower_balanced_error_does_not_override_frontal_preservation_gate(self):
        """Reject a capacity that improves aggregate validation while degrading frontal error beyond 5%.
        Keep the failed candidate's typed evaluation and never execute a final personalized refit.
        """
        views, runs, score = self.fixture()
        policy = FitPolicy(identity_candidates=(1,))
        report = ReconstructionReport(policy, ReconstructionInputs(len(views)))
        regressed = score(0.8)
        regressed["per_initial_view_category"]["front"] = 1.06
        with (
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.validation_indices",
                return_value=[1],
            ),
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.fit",
                side_effect=runs,
            ) as fit,
            patch(
                "ailias.head_reconstruction.geometry.identity_fitting.evaluation_score",
                side_effect=[score(1), regressed],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "criterion failed"):
                fit_shared_head_identity(object(), views, policy, report)
        self.assertEqual(fit.call_count, 4)
        self.assertFalse(report.candidates[0].frontal_preserved)
