"""Connect retained feature records to synthetic CPU fitting without neural/native runtimes."""

import json
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ailias.head_reconstruction.evidence.feature_observations import observations_from_face_features
from ailias.head_reconstruction.geometry.feature_reconstruction import (
    HeadReconstructionError, reconstruct_head_from_features,
)
from ailias.head_reconstruction.geometry.optimization import prepare_views, project
from ailias.head_reconstruction.settings import FitPolicy
from ailias.imageprocessing.face_features import FaceFeatures, FaceQuality


MODULE = "ailias.head_reconstruction.geometry.feature_reconstruction"


class FeatureReconstructionTests(unittest.TestCase):
    def setUp(self):
        """Create a small supplied head prior and multi-view synthetic landmark evidence.
        Use no downloaded templates, user photographs, neural networks or hardware probes.
        """
        x, y = np.meshgrid(np.linspace(-.07, .07, 26), np.linspace(-.09, .09, 18))
        points = np.column_stack((x.ravel(), y.ravel(), (.03 - .012 * (x / .07)**2
                                                        - .008 * (y / .09)**2).ravel()))
        cells = np.arange(468).reshape(18, 26)[:-1, :-1].ravel()
        triangles = np.concatenate((
            np.column_stack((cells, cells + 1, cells + 26)),
            np.column_stack((cells + 1, cells + 27, cells + 26)),
        ))
        identity = np.zeros((1, 468, 3))
        identity[0, :, 0] = .012 * np.cos(points[:, 1] * 25)
        expressions = np.zeros((1, 468, 3))
        expressions[0, :, 1] = .01 * (points[:, 1] < -.025)
        self.model = SimpleNamespace(
            points=points, center=np.zeros(3), vertices=points.copy(),
            triangles=triangles,
            normals=np.tile([0., 0., 1.], (468, 1)), supported=np.ones(468, dtype=bool),
            regions=np.zeros(468, dtype=int), region_names=["face"],
            identity_points=identity, expression_points=expressions,
            neutral=lambda coefficients: points + np.einsum("i,ivc->vc", coefficients, identity),
        )
        self.policy = FitPolicy(
            device="cpu", identity_candidates=(1,), expression_modes=(1, 1, 1),
            identity_regularization=1e-8, expression_regularization=1e-8, max_nfev=45,
        )
        cameras = np.array([
            [3.1, 0, 0, 0, 0, .7, 0], [3.1, .4, 0, .02, 0, .65, .1],
            [3.1, -.4, 0, -.02, 0, .75, -.1], [3.0, .2, 0, .01, .01, .7, 0],
            [3.2, -.2, 0, -.01, -.01, .7, 0], [3.15, .1, 0, .01, 0, .72, .05],
        ])
        expressions_per_view = np.array([[-.7], [.8], [.1], [.2], [-.3], [.5]])
        world = points[None] + .9 * identity[0] + np.einsum(
            "ni,ivc->nvc", expressions_per_view, expressions)
        targets, _ = project(world, cameras, self.model.center)
        self.features = {}
        for index, target in enumerate(targets):
            size = np.array([1200, 900] if index % 2 == 0 else [900, 1200])
            xy = np.concatenate((target, target[:10])) * max(size) + size / 2
            features = FaceFeatures(
                source_size=tuple(size.tolist()),
                landmarks={"source_xy": xy.tolist(), "on_head": [True] * 478},
                orientation={"reason": "missing_or_invalid_pose", "calibrated": False},
                quality=FaceQuality(reasons=["non_neutral_expression"]),
            )
            self.features[f"view-{index}"] = FaceFeatures.from_dict(json.loads(str(features)))

    def test_serialized_features_reach_real_numerical_fit_and_recover_neutral_shape_without_reinference(self):
        """Run the actual camera/identity/expression solver on a synthetic, explicitly supplied prior.
        Recover shared deformation while keeping predicted per-view expressions out of the neutral mesh.
        """
        before = {key: value.to_dict() for key, value in self.features.items()}
        with patch("ailias.imageprocessing.dense.DenseDetector",
                   side_effect=AssertionError("Landmark extraction was rerun")), \
                patch("PIL.Image.open", side_effect=AssertionError("Pixels were reloaded")):
            result = reconstruct_head_from_features(self.model, self.features, self.policy)
        self.assertEqual(result["report"]["status"], "awaiting_human_likeness_review")
        self.assertEqual(result["report"]["excluded"], [])
        self.assertEqual(len(result["report"]["prepared_view_ids"]), 6)
        self.assertTrue(result["report"]["mesh_quality"]["passed"])
        self.assertAlmostEqual(result["fit"]["identity"][0], .9, delta=.2)
        np.testing.assert_allclose(result["vertices"], self.model.neutral(result["fit"]["identity"]))
        np.testing.assert_array_equal(result["triangles"], self.model.triangles)
        self.assertEqual({key: value.to_dict() for key, value in self.features.items()}, before)

    def test_non_square_pixel_grids_are_normalized_using_the_retained_dimensions(self):
        """Normalize each target with its own width/height, without a source image or guessed crop size.
        Preserve the original point association when aspect ratios differ across photos.
        """
        observations = observations_from_face_features(self.features)
        views, excluded = prepare_views(
            self.model, observations, {o["face_id"]: {} for o in observations})
        self.assertFalse(excluded)
        for view in views:
            features = self.features[view["face_id"]]
            size = np.asarray(features.source_size)
            expected = (np.asarray(features.landmarks["source_xy"])[:468] - size / 2) / max(size)
            np.testing.assert_allclose(view["target"], expected)

    def test_new_path_retains_extreme_pose_but_legacy_default_rejection_is_unchanged(self):
        """Keep numerically supported upper/side views in the new feature-based connection.
        Preserve the old pipeline's explicit pose policy without changing image-grouping decisions.
        """
        extreme = {"yaw_deg": 120, "pitch_deg": 70, "category": "right_profile"}
        observations = observations_from_face_features(self.features)
        entries = {observation["face_id"]: {} for observation in observations}
        with patch("ailias.head_reconstruction.geometry.optimization.dense_view", return_value=extreme):
            legacy_report = {}
            with self.assertRaisesRegex(ValueError, "five reliable"):
                prepare_views(self.model, observations, entries, report=legacy_report)
            self.assertTrue(all(o["reason"] == "extreme_or_ambiguous_dense_pose"
                                for o in legacy_report["excluded"]))
            fitted = {"identity": np.array([0.]), "report": {"fixture": True}}
            with patch(MODULE + ".fit_multiview_identity", return_value=fitted) as fit:
                result = reconstruct_head_from_features(self.model, self.features, self.policy)
        self.assertEqual(len(fit.call_args.args[1]), 6)
        self.assertEqual(result["report"]["excluded"], [])

    def test_missing_dimensions_and_camera_failure_remain_in_failure_report(self):
        """Expose every rejected view even when too few observations remain to attempt identity fitting.
        Never invent dimensions for older features or silently return a generic/template head.
        """
        self.features["view-0"] = FaceFeatures(landmarks=self.features["view-0"].landmarks)
        with patch("ailias.head_reconstruction.geometry.optimization.initialize_camera",
                   side_effect=ValueError("fixture camera failure")), \
                patch(MODULE + ".fit_multiview_identity") as fit:
            with self.assertRaises(HeadReconstructionError) as raised:
                reconstruct_head_from_features(self.model, self.features, self.policy)
        report = raised.exception.report
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["prepared_view_ids"], [])
        reasons = {o["face_id"]: o["reason"] for o in report["excluded"]}
        self.assertEqual(reasons.pop("view-0"), "missing_or_invalid_source_size")
        self.assertEqual(list(reasons.values()), ["fixture camera failure"] * 5)
        fit.assert_not_called()

    def test_foreground_support_is_still_required_even_when_orientation_and_quality_do_not_gate_views(self):
        """Reject geometrically unsupported observations without reverting to angle/quality rejection.
        Retain each numerical failure and avoid starting an underconstrained identity fit.
        """
        features = {
            key: replace(value, landmarks={**value.landmarks, "on_head": [False] * 478})
            for key, value in self.features.items()
        }
        with patch(MODULE + ".fit_multiview_identity") as fit:
            with self.assertRaises(HeadReconstructionError) as raised:
                reconstruct_head_from_features(self.model, features, self.policy)
        exclusions = raised.exception.report["excluded"]
        self.assertEqual(len(exclusions), 6)
        self.assertTrue(all(item["reason"] == "Insufficient supported dense landmarks" for item in exclusions))
        fit.assert_not_called()

    def test_identity_quality_failure_and_invalid_mesh_are_not_published(self):
        """Keep existing held-out identity gates and reject nonfinite or collapsed neutral meshes.
        Preserve the underlying exception plus diagnostics instead of substituting an unfitted head.
        """
        fitted = {"identity": np.array([0.]), "report": {}}
        with patch(MODULE + ".fit_multiview_identity", side_effect=ValueError("fixture identity gate")):
            with self.assertRaisesRegex(HeadReconstructionError, "identity gate") as raised:
                reconstruct_head_from_features(self.model, self.features, self.policy)
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        for mesh, reason in (
            (np.full_like(self.model.vertices, np.nan), "Non-finite"),
            (np.zeros_like(self.model.vertices), "collapses"),
        ):
            with self.subTest(reason=reason), \
                    patch(MODULE + ".fit_multiview_identity", return_value=fitted), \
                    patch.object(self.model, "neutral", return_value=mesh):
                with self.assertRaisesRegex(HeadReconstructionError, reason):
                    reconstruct_head_from_features(self.model, self.features, self.policy)
