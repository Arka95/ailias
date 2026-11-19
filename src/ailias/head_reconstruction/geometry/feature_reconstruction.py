"""Coordinate typed, stage-scoped head reconstruction from retained facial evidence."""

from collections.abc import Mapping, Sequence

import numpy as np

from ailias.head_reconstruction.evidence.feature_observations import (
    observations_from_face_features,
)
from ailias.head_reconstruction.geometry.identity_fitting import (
    fit_shared_head_identity,
)
from ailias.head_reconstruction.geometry.mesh_quality import assess_mesh_deformation
from ailias.head_reconstruction.geometry.optimization import prepare_views
from ailias.head_reconstruction.reconstruction_data import (
    ExcludedHeadView,
    HeadFittingResult,
    HeadMesh,
    HeadReconstructionResult,
    HeadViewObservation,
    ReconstructionFailure,
    ReconstructionInputs,
    ReconstructionReport,
)
from ailias.head_reconstruction.settings import FitPolicy
from ailias.imageprocessing.face_features import FaceFeatures


class HeadReconstructionError(ValueError):
    """Expose numerical/evidence failure together with its typed per-view diagnostic report.
    Preserve the original exception as the cause rather than silently publishing a generic head.
    """

    def __init__(self, message: str, report: ReconstructionReport):
        """Retain completed reconstruction diagnostics without writing failure artifacts.
        Let callers inspect evidence exclusions and candidate rejection through the ordinary exception path.
        """
        super().__init__(message)
        self.report = report


def reconstruct_head_from_features(
    model,
    features_by_view: Mapping[str, FaceFeatures],
    policy: FitPolicy,
) -> HeadReconstructionResult:
    """Fit and validate one neutral head using existing feature evidence and an explicitly supplied prior.
    Delegate observation conversion, camera preparation, identity fitting and mesh construction to scoped stages.
    """
    report = ReconstructionReport(policy, ReconstructionInputs(len(features_by_view)))
    try:
        observations = observations_from_face_features(features_by_view)
        report.inputs.landmarks_ready = sum(o.status == "ready" for o in observations)
        views = prepare_head_views(model, observations, report)
        fitted = fit_shared_head_identity(model, views, policy, report)
        mesh = construct_neutral_head_mesh(model, fitted, report)
    except (ValueError, RuntimeError) as error:
        report.status = "failed"
        report.failure = ReconstructionFailure(type(error).__name__, str(error))
        raise HeadReconstructionError(str(error), report) from error
    report.status = "awaiting_human_likeness_review"
    return HeadReconstructionResult(mesh, fitted, observations, report)


def prepare_head_views(
    model,
    observations: Sequence[HeadViewObservation],
    report: ReconstructionReport,
) -> list[dict]:
    """Prepare numerical cameras/targets from typed observations without quality or angle rejection.
    Retain per-view failure diagnostics even when the existing five-view/support requirements cannot be met.
    """
    solver_records = [_solver_observation(observation) for observation in observations]
    diagnostics = {}
    entries = {observation.view_id: {} for observation in observations}
    try:
        views, _ = prepare_views(
            model,
            solver_records,
            entries,
            reject_extreme_pose=False,
            report=diagnostics,
        )
    finally:
        report.prepared_view_ids = diagnostics.get("prepared_view_ids", [])
        report.excluded = [
            ExcludedHeadView(item["face_id"], item["reason"])
            for item in diagnostics.get("excluded", [])
        ]
    return views


def _solver_observation(observation: HeadViewObservation) -> dict:
    """Adapt typed evidence to the numerical backend's existing array/identifier contract.
    Keep legacy field names private to the fitting boundary rather than exposing them as public output.
    """
    return {
        "face_id": observation.view_id,
        "status": observation.status,
        "reason": observation.reason,
        "source_size": observation.source_size,
        "source_xy": observation.source_xy,
        "on_head": observation.on_head,
        "eye_span_source": observation.eye_span_source,
    }


def construct_neutral_head_mesh(
    model,
    fitted: HeadFittingResult,
    report: ReconstructionReport,
) -> HeadMesh:
    """Build neutral geometry from shared identity only and apply the existing surface-deformation gates.
    Exclude per-view expression, reject nonfinite/invalid meshes and leave all exports to the caller.
    """
    vertices = model.neutral(fitted.parameters.identity_coefficients)
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite neutral head")
    report.mesh_quality = assess_mesh_deformation(
        model.vertices, vertices, model.triangles
    )
    if not report.mesh_quality.passed:
        raise ValueError(
            "Head deformation collapses, flips or excessively stretches the template"
        )
    return HeadMesh(vertices, model.triangles.copy())
