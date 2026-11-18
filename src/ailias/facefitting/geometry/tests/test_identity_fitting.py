"""Shared identity selection and original-view ordering, without running a fitter."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ailias.facefitting.geometry.identity_fitting import fit_multiview_identity


class SharedIdentityTests(unittest.TestCase):
    def fixture(self):
        views = [{"face_id": str(index)} for index in range(3)]
        runs = [
            {"per_view": np.zeros((2, 2))},
            {"per_view": np.zeros((1, 2)), "report": {}},
            {"identity": np.array([.25]), "per_view": np.array([[10., 11.], [30., 31.]])},
            {"per_view": np.array([[20., 21.]]), "report": {}},
            {"final_fit": True},
        ]
        score = lambda value: {
            "view_balanced_mean_error": value, "per_initial_view_category": {"front": value}}
        return views, runs, score

    def test_final_fit_restores_training_and_heldout_cameras_to_original_view_order(self):
        views, runs, score = self.fixture()
        report = {"inputs": {}}
        with patch("ailias.facefitting.geometry.identity_fitting.validation_indices", return_value=[1]), \
                patch("ailias.facefitting.geometry.identity_fitting.fit", side_effect=runs) as fit, \
                patch("ailias.facefitting.geometry.identity_fitting.evaluation_score",
                      side_effect=[score(1), score(.8)]):
            result = fit_multiview_identity(object(), views, SimpleNamespace(identity_candidates=[1]), report)
        self.assertEqual(result, {"final_fit": True})
        np.testing.assert_array_equal(fit.call_args.kwargs["initial"],
                                      [.25, 10., 11., 20., 21., 30., 31.])
        self.assertEqual(report["inputs"]["training"], ["0", "2"])
        self.assertEqual(report["inputs"]["heldout"], ["1"])

    def test_insufficient_heldout_improvement_does_not_run_a_final_personalized_fit(self):
        views, runs, score = self.fixture()
        with patch("ailias.facefitting.geometry.identity_fitting.validation_indices", return_value=[1]), \
                patch("ailias.facefitting.geometry.identity_fitting.fit", side_effect=runs) as fit, \
                patch("ailias.facefitting.geometry.identity_fitting.evaluation_score",
                      side_effect=[score(1), score(.995)]):
            with self.assertRaisesRegex(ValueError, "criterion failed"):
                fit_multiview_identity(object(), views, SimpleNamespace(identity_candidates=[1]), {"inputs": {}})
        self.assertEqual(fit.call_count, 4)
