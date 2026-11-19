"""Reconstruct a neutral head from retained facial evidence, without image inference or native editors."""

from collections.abc import Mapping

import numpy as np

from ailias.head_reconstruction.evidence.feature_observations import observations_from_face_features
from ailias.head_reconstruction.geometry.identity_fitting import fit_multiview_identity
from ailias.head_reconstruction.geometry.mesh_quality import surface_quality
from ailias.head_reconstruction.geometry.optimization import prepare_views
from ailias.head_reconstruction.settings import FitPolicy
from ailias.imageprocessing.face_features import FaceFeatures


class HeadReconstructionError(ValueError):
    """Expose numerical/evidence failure together with its per-view diagnostic report.
    Preserve the original exception as the cause rather than silently publishing a generic head.
    """

    def __init__(self, message: str, report: dict):
        """Retain the current fitting diagnostics without writing failure artifacts.
        Let callers inspect exclusions and candidate rejection through the ordinary exception path.
        """
        super().__init__(message)
        self.report = report


def reconstruct_head_from_features(
    model, features_by_view: Mapping[str, FaceFeatures], policy: FitPolicy,
) -> dict:
    """Fit one shared identity with per-view cameras/expressions from existing facial predictions.
    Return neutral vertices, topology, parameters and diagnostics; require an explicitly loaded model/policy.
    """
    report = {
        "status": "fitting", "policy": policy.record(),
        "inputs": {"observations": len(features_by_view)},
        "limitations": [
            "Predicted landmarks constrain a supplied head prior, not a measured 3D scan.",
            "Scale, focal length and hidden anatomy remain uncalibrated or prior-derived.",
            "Alpha and template normals are support heuristics, not per-point visibility ground truth.",
            "Photo colours, orientation estimates and MediaPipe blendshapes are not geometry coefficients.",
            "Reprojection and surface checks do not certify visual likeness or self-intersection freedom.",
        ],
    }
    try:
        observations = observations_from_face_features(features_by_view)
        report["inputs"]["landmarks_ready"] = sum(o["status"] == "ready" for o in observations)
        entries = {observation["face_id"]: {} for observation in observations}
        views, _ = prepare_views(
            model, observations, entries, reject_extreme_pose=False, report=report)
        result = fit_multiview_identity(model, views, policy, report)
        vertices = model.neutral(result["identity"])
        if not np.isfinite(vertices).all():
            raise ValueError("Non-finite neutral head")
        report["mesh_quality"] = surface_quality(model.vertices, vertices, model.triangles)
        if not report["mesh_quality"]["passed"]:
            raise ValueError("Head deformation collapses, flips or excessively stretches the template")
    except (ValueError, RuntimeError) as error:
        report.update(status="failed", error_type=type(error).__name__, reason=str(error))
        raise HeadReconstructionError(str(error), report) from error
    report.update(status="awaiting_human_likeness_review", fit=result["report"])
    return {
        "vertices": vertices, "triangles": model.triangles.copy(),
        "fit": result, "observations": observations, "report": report,
    }
