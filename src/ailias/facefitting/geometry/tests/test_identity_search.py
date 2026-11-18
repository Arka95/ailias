"""Search all configured capacities and retain failures without silently limiting input views."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from ailias.facefitting.configuration import FitPolicy
from ailias.facefitting.geometry.identity_search import IdentitySearchPolicy, search_identity_fit


class IdentitySearchTests(unittest.TestCase):
    def test_identity_and_expression_priors_are_searched_independently(self):
        views = [{"face_id": str(i), "view": {"category": "front"}} for i in range(6)]
        policy = FitPolicy(identity_candidates=(1,), expression_modes=(1, 1, 1))
        search = IdentitySearchPolicy((1,), ((1, 1, 1),), (.1, 1), (3,), (.1, 10))
        retained = []

        def fake_fit(model, selected, configuration, modes, fixed_identity=None, initial=None):
            return {"identity": np.ones(modes), "per_view": np.zeros((len(selected), 8)),
                    "report": {"objective": .1, "views": [{"face_id": view["face_id"],
                                         "weighted_rms_outer_eye_span_error": .1} for view in selected]}}

        report = {"inputs": {}}
        with patch("ailias.facefitting.geometry.identity_search.fit", side_effect=fake_fit):
            search_identity_fit(lambda modes: None, views, policy, search, report,
                                lambda record, candidate: retained.append(record["policy"]))
        ratios = {(round(p["identity_regularization"]/policy.identity_regularization, 2),
                   round(p["expression_regularization"]/policy.expression_regularization, 2))
                  for p in retained}
        self.assertEqual(ratios, {(.1, .1), (.1, 10), (1, .1), (1, 10)})
        self.assertEqual(len(report["search"]["matched_identity_comparison"]), 1)

    def test_all_candidates_are_evaluated_and_best_identity_restores_all_view_order(self):
        views = [{"face_id": str(i), "view": {"category": "front"}} for i in range(6)]
        policy = FitPolicy(identity_candidates=(1, 2), expression_modes=(1, 1, 1))
        search = IdentitySearchPolicy((1, 2), ((1, 1, 1),), (1,), (3,))
        calls, retained = [], []

        def fake_fit(model, selected, configuration, modes, fixed_identity=None, initial=None):
            calls.append((selected, modes, initial))
            return {"identity": np.ones(modes), "per_view": np.array([
                [float(view["face_id"])]*8 for view in selected]),
                "report": {"objective": .2/(modes+1), "views": [{"face_id": view["face_id"],
                                     "weighted_rms_outer_eye_span_error": .2/(modes+1)}
                                    for view in selected]}}

        report = {"inputs": {}}
        with patch("ailias.facefitting.geometry.identity_search.fit", side_effect=fake_fit):
            model, result, selected = search_identity_fit(
                lambda modes: "model", views, policy, search, report,
                lambda record, candidate: retained.append(record.copy()))
        self.assertEqual(len(retained), 2)
        self.assertEqual(report["search"]["selected_index"], 1)
        self.assertEqual(len(result["identity"]), 2)
        comparison = report["search"]["matched_identity_comparison"]
        self.assertEqual([item["identity_modes"] for item in comparison], [1, 2])
        self.assertAlmostEqual(comparison[0]["training_score"]["view_balanced_rms"], .1)
        self.assertAlmostEqual(comparison[1]["heldout_score"]["view_balanced_rms"], .2/3)
        self.assertEqual(len(calls[-1][0]), len(views))
        np.testing.assert_array_equal(calls[-1][2][2:].reshape(6, 8)[:, 0], np.arange(6))

    def test_full_lower_face_capacity_and_wider_bounds_are_supported(self):
        policy = FitPolicy(expression_modes=(100, 100, 150), expression_coefficient_limit=6)
        self.assertEqual(policy.expression_modes, (100, 100, 150))
        for expression in ((101, 1, 1), (1, 101, 1), (1, 1, 151)):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                replace(policy, expression_modes=expression)

    def test_failed_candidate_is_recorded_and_other_capacity_is_still_tried(self):
        views = [{"face_id": str(i), "view": {"category": "front"}} for i in range(6)]
        policy = FitPolicy(identity_candidates=(1, 2), expression_modes=(1, 1, 1))
        search = IdentitySearchPolicy((1, 2), ((1, 1, 1),), (1,), (3,))
        retained = []

        def fake_fit(model, selected, configuration, modes, fixed_identity=None, initial=None):
            if modes == 1:
                raise ValueError("invalid candidate")
            return {"identity": np.ones(modes), "per_view": np.zeros((len(selected), 8)),
                    "report": {"objective": .1, "views": [{"face_id": view["face_id"],
                                         "weighted_rms_outer_eye_span_error": .1} for view in selected]}}

        report = {"inputs": {}}
        with patch("ailias.facefitting.geometry.identity_search.fit", side_effect=fake_fit):
            search_identity_fit(lambda modes: None, views, policy, search, report,
                                lambda record, candidate: retained.append(record.copy()))
        self.assertEqual([record["status"] for record in retained], ["failed", "completed"])
        self.assertEqual(report["search"]["failed_candidates"], 1)

    def test_larger_capacity_also_starts_from_the_smaller_fit_without_using_heldout_targets(self):
        views = [{"face_id": str(i), "view": {"category": "front"}} for i in range(6)]
        policy = FitPolicy(identity_candidates=(1, 2), expression_modes=(1, 1, 1))
        search = IdentitySearchPolicy((1, 2), ((1, 1, 1),), (1,), (3,))

        def fake_fit(model, selected, configuration, modes, fixed_identity=None, initial=None):
            if fixed_identity is not None and len(fixed_identity):
                error = float(fixed_identity[0])
            elif modes == 1:
                error = .1
            elif modes == 2:
                error = .09 if initial is not None and initial[0] in (.1, .09) else .2
                if initial is not None and initial[0] == .1:
                    self.assertEqual([v["face_id"] for v in selected], ["0", "1", "2", "3", "5"])
                    self.assertEqual(initial[1], 0)
            else:
                error = .3
            return {"identity": np.full(modes, error), "per_view": np.zeros((len(selected), 8)),
                    "report": {"objective": error, "views": [
                        {"face_id": view["face_id"], "weighted_rms_outer_eye_span_error": error}
                        for view in selected]}}

        report = {"inputs": {}}
        with patch("ailias.facefitting.geometry.identity_search.fit", side_effect=fake_fit):
            search_identity_fit(lambda modes: None, views, policy, search, report,
                                lambda record, candidate: None)
        selected = report["search"]["candidates"][report["search"]["selected_index"]]
        self.assertEqual(selected["identity_modes"], 2)
        self.assertEqual(selected["selected_training_initialization"], "lower_capacity")
        self.assertEqual([v["objective"] for v in selected["training_initializations"]], [.2, .09])
