"""Select shared head identity using held-out views and separate per-view expressions."""

from dataclasses import dataclass

import numpy as np

from ailias.head_reconstruction.geometry.fitting_results import (
    fitting_result_from_solver,
)
from ailias.head_reconstruction.geometry.optimization import (
    evaluation_score,
    fit,
    validation_indices,
)
from ailias.head_reconstruction.reconstruction_data import (
    FitEvaluation,
    HeadFittingResult,
    IdentityCandidateEvaluation,
    ReconstructionInputs,
    ReconstructionReport,
)


# Preserve the existing 1% held-out improvement and 5% frontal non-regression gates.
MIN_HELDOUT_IDENTITY_IMPROVEMENT = 0.01
MAX_FRONTAL_ERROR_REGRESSION = 0.05


@dataclass(frozen=True)
class _CandidateFit:
    """Keep the selected training/validation numerical solutions together with their evaluation.
    Internal backend records do not escape through the typed reconstruction API.
    """

    training: dict
    validation: dict
    score: FitEvaluation


def fit_shared_head_identity(
    model, views, policy, report: ReconstructionReport
) -> HeadFittingResult:
    """Fit one shared identity and return typed parameters/metrics after held-out candidate selection.
    Reuse existing CPU/CUDA solvers without mixing image inference, mesh creation or file export into fitting.
    """
    return fitting_result_from_solver(
        _fit_identity_parameters(model, views, policy, report)
    )


def _fit_identity_parameters(
    model, views, policy, report: ReconstructionReport
) -> dict:
    """Separate view splitting, generic reference fitting, candidate selection and final all-view refit.
    Preserve the original view order when recombining training and validation camera/expression parameters.
    """
    heldout = validation_indices(views)
    training = [index for index in range(len(views)) if index not in heldout]
    train, validation = (
        [views[index] for index in indices] for indices in (training, heldout)
    )
    report.inputs.training_view_ids = [view["face_id"] for view in train]
    report.inputs.validation_view_ids = [view["face_id"] for view in validation]
    baseline_train, baseline_validation, baseline_score = _fit_generic_reference(
        model, train, validation, policy
    )
    report.baseline_score = baseline_score
    best = _select_identity_candidate(
        model, train, validation, policy, baseline_train, baseline_validation, report
    )
    if (
        best is None
        or best.score.view_balanced_mean_error
        >= baseline_score.view_balanced_mean_error
        * (1 - MIN_HELDOUT_IDENTITY_IMPROVEMENT)
    ):
        raise ValueError(
            "Existing shared-identity fit criterion failed; no personalized head published"
        )
    report.heldout_improvement = (
        1
        - best.score.view_balanced_mean_error / baseline_score.view_balanced_mean_error
    )
    initial = _restore_all_view_parameters(best, len(views), training, heldout)
    return fit(model, views, policy, len(best.training["identity"]), initial=initial)


def _fit_generic_reference(
    model, training, validation, policy
) -> tuple[dict, dict, FitEvaluation]:
    """Fit cameras/expressions with zero identity deformation for training and held-out reference views.
    Establish the same category-balanced baseline used to evaluate every personal-shape candidate.
    """
    baseline_train = fit(model, training, policy, 0, fixed_identity=np.zeros(0))
    baseline_validation = fit(model, validation, policy, 0, fixed_identity=np.zeros(0))
    score = FitEvaluation(**evaluation_score(baseline_validation["report"], validation))
    return baseline_train, baseline_validation, score


def _select_identity_candidate(
    model,
    training,
    validation,
    policy,
    baseline_train,
    baseline_validation,
    report: ReconstructionReport,
) -> _CandidateFit | None:
    """Evaluate configured identity capacities without allowing held-out frontal accuracy to regress.
    Retain each completed candidate diagnostic, including when a later solver invocation fails.
    """
    best = None
    for modes in policy.identity_candidates:
        candidate = _evaluate_identity_candidate(
            model,
            training,
            validation,
            policy,
            modes,
            baseline_train,
            baseline_validation,
        )
        front = candidate.score.per_initial_view_category.get("front")
        base_front = report.baseline_score.per_initial_view_category.get("front")
        preserved = (
            front is None
            or base_front is None
            or front <= base_front * (1 + MAX_FRONTAL_ERROR_REGRESSION)
        )
        report.candidates.append(
            IdentityCandidateEvaluation(modes, candidate.score, preserved)
        )
        if preserved and (
            best is None
            or candidate.score.view_balanced_mean_error
            < best.score.view_balanced_mean_error
        ):
            best = candidate
    return best


def _evaluate_identity_candidate(
    model,
    training,
    validation,
    policy,
    modes,
    baseline_train,
    baseline_validation,
) -> _CandidateFit:
    """Fit identity on training views, then hold it fixed while fitting validation cameras/expressions.
    Compare against the original validation categories rather than candidate-dependent view relabeling.
    """
    initial = np.r_[np.zeros(modes), baseline_train["per_view"].ravel()]
    candidate = fit(model, training, policy, modes, initial=initial)
    tested = fit(
        model,
        validation,
        policy,
        modes,
        fixed_identity=candidate["identity"],
        initial=baseline_validation["per_view"].ravel(),
    )
    return _CandidateFit(
        candidate,
        tested,
        FitEvaluation(**evaluation_score(tested["report"], validation)),
    )


def _restore_all_view_parameters(
    candidate: _CandidateFit, count: int, training, heldout
) -> np.ndarray:
    """Reassemble camera/expression rows by their original view positions before the final refit.
    Keep the chosen shared identity separate from per-view nuisance parameters.
    """
    per_view = np.empty((count, candidate.training["per_view"].shape[1]))
    per_view[training], per_view[heldout] = (
        candidate.training["per_view"],
        candidate.validation["per_view"],
    )
    return np.r_[candidate.training["identity"], per_view.ravel()]


def fit_multiview_identity(model, views, policy, report):
    """Preserve the existing dictionary-based pipeline API while sharing the scoped fitting implementation.
    Keep legacy callers and persisted report keys unchanged; new consumers use fit_shared_head_identity().
    """
    diagnostics = ReconstructionReport(policy, ReconstructionInputs(len(views)))
    try:
        return _fit_identity_parameters(model, views, policy, diagnostics)
    finally:
        _copy_legacy_diagnostics(diagnostics, report)


def _copy_legacy_diagnostics(diagnostics: ReconstructionReport, report: dict) -> None:
    """Translate completed typed selection diagnostics into the existing legacy manifest fields.
    Preserve extra caller-owned input/report fields rather than replacing their surrounding dictionaries.
    """
    report["inputs"].update(
        training=diagnostics.inputs.training_view_ids,
        heldout=diagnostics.inputs.validation_view_ids,
    )
    if diagnostics.baseline_score is not None:
        report["baseline_score"] = diagnostics.baseline_score.to_dict()
        report["candidates"] = [
            {
                "modes": candidate.identity_modes,
                "score": candidate.score.to_dict(),
                "frontal_preserved": candidate.frontal_preserved,
            }
            for candidate in diagnostics.candidates
        ]
    if diagnostics.heldout_improvement is not None:
        report["heldout_improvement"] = diagnostics.heldout_improvement
