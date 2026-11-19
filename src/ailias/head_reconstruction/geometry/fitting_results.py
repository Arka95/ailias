"""Convert numerical-backend records into the typed public head-fitting result."""

from ailias.head_reconstruction.reconstruction_data import (
    FittedHeadView,
    FittingRuntime,
    HeadFitParameters,
    HeadFitSummary,
    HeadFittingResult,
)


_SUMMARY_KEYS = {
    "identity_modes",
    "expression_modes",
    "views",
    "median_outer_eye_span_error",
}
_RUNTIME_KEYS = {
    "backend",
    "device",
    "seconds",
    "nfev",
    "converged",
    "termination",
    "objective",
}


def fitting_result_from_solver(result: dict) -> HeadFittingResult:
    """Translate the existing CPU/CUDA array contract at the reconstruction boundary.
    Preserve numerical precision and view association without changing either solver's objective.
    """
    metrics = result["report"]
    summary = HeadFitSummary(
        identity_modes=metrics["identity_modes"],
        expression_modes=metrics["expression_modes"],
        views=[_fitted_view_from_solver(view) for view in metrics["views"]],
        runtime=_runtime_from_solver(metrics),
        median_outer_eye_span_error=metrics["median_outer_eye_span_error"],
    )
    parameters = HeadFitParameters(
        identity_coefficients=result["identity"],
        cameras_and_expressions=result["per_view"],
        projected_landmarks=result["projected"],
        optimizer_parameters=result["parameters"],
    )
    return HeadFittingResult(parameters, summary)


def _fitted_view_from_solver(view: dict) -> FittedHeadView:
    """Rename the legacy face identifier to its actual single-person view responsibility.
    Retain all reprojection metrics and the uncalibrated camera-orientation diagnostic.
    """
    return FittedHeadView(
        view_id=view["face_id"],
        orientation=dict(view["view"]),
        **{
            name: value
            for name, value in view.items()
            if name not in {"face_id", "view"}
        },
    )


def _runtime_from_solver(metrics: dict) -> FittingRuntime:
    """Separate common solver measurements from backend-specific hardware/runtime provenance.
    Retain optional CUDA metadata without requiring Torch during ordinary record conversion.
    """
    return FittingRuntime(
        backend=metrics["backend"],
        device=metrics["device"],
        seconds=metrics["seconds"],
        evaluations=metrics["nfev"],
        converged=metrics["converged"],
        termination=metrics["termination"],
        objective=metrics["objective"],
        backend_details={
            key: value
            for key, value in metrics.items()
            if key not in _SUMMARY_KEYS | _RUNTIME_KEYS
        },
    )
