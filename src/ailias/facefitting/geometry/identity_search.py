"""Search explicit model capacities and priors against the same held-out observations."""

from dataclasses import dataclass, replace
import itertools
import json
import math

import numpy as np

from ailias.facefitting.configuration import FitPolicy
from ailias.facefitting.geometry.optimization import fit, validation_indices


@dataclass(frozen=True)
class IdentitySearchPolicy:
    identity_candidates: tuple
    expression_candidates: tuple
    identity_regularization_factors: tuple
    expression_limits: tuple
    expression_regularization_factors: tuple = (1.0,)

    def __post_init__(self):
        if not all((self.identity_candidates, self.expression_candidates,
                    self.identity_regularization_factors, self.expression_limits,
                    self.expression_regularization_factors)):
            raise ValueError("Fit search dimensions must be nonempty")
        for expression in self.expression_candidates:
            FitPolicy(identity_candidates=self.identity_candidates, expression_modes=expression)
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0
               for v in (*self.identity_regularization_factors, *self.expression_regularization_factors,
                         *self.expression_limits)):
            raise ValueError("Search regularization factors and expression limits must be finite and positive")


def load_search_policy(path):
    return IdentitySearchPolicy(**json.loads(path.read_text(encoding="utf-8")))


def residual_score(result, views):
    categories = {view["face_id"]: view["view"]["category"] for view in views}
    errors = {}
    for metric in result["report"]["views"]:
        errors.setdefault(categories[metric["face_id"]], []).append(
            metric["weighted_rms_outer_eye_span_error"])
    per_category = {name: float(np.mean(values)) for name, values in errors.items()}
    if not per_category or not all(math.isfinite(value) for value in per_category.values()):
        raise ValueError("Held-out residuals must be nonempty and finite")
    return {"view_balanced_rms": float(np.mean(list(per_category.values()))),
            "per_initial_view_category": per_category}


def search_identity_fit(model_factory, views, policy, search, report, retain_candidate):
    """Keep every prepared view; batch size changes memory use, not the evidence set."""
    heldout = validation_indices(views)
    training = [index for index in range(len(views)) if index not in heldout]
    train, validation = ([views[index] for index in indices] for indices in (training, heldout))
    report["inputs"].update(training=[view["face_id"] for view in train],
                            heldout=[view["face_id"] for view in validation])
    results = report["search"] = {
        "selection_metric": "view-category-balanced weighted RMS of all supported held-out landmarks",
        "planned_candidates": (len(search.identity_candidates)*len(search.expression_candidates)
                               * len(search.identity_regularization_factors)*len(search.expression_limits)
                               * len(search.expression_regularization_factors)),
        "candidates": [],
        "heldout_scope": "identity held out; camera/expression optimized per validation view",
    }
    best = None
    for expression in search.expression_candidates:
        model = model_factory(expression)
        references = {}
        for identity_factor, expression_factor, limit in itertools.product(
            search.identity_regularization_factors, search.expression_regularization_factors,
            search.expression_limits,
        ):
            candidate_policy = replace(
                policy, expression_modes=tuple(expression),
                identity_regularization=policy.identity_regularization*identity_factor,
                expression_regularization=policy.expression_regularization*expression_factor,
                expression_coefficient_limit=limit,
            )
            reference_key = (expression_factor, limit)
            if reference_key not in references:
                reference = reference_validation = None
                generic_score, generic_error = None, None
                try:
                    reference = fit(model, train, candidate_policy, 0, fixed_identity=np.zeros(0))
                    reference_validation = fit(model, validation, candidate_policy, 0, fixed_identity=np.zeros(0))
                    generic_score = residual_score(reference_validation, validation)
                except (ValueError, RuntimeError) as error:
                    generic_error = {"type": type(error).__name__, "message": str(error)}
                # Identity regularization has no residual when the identity vector is empty.
                if generic_error is None:
                    references[reference_key] = reference, reference_validation, generic_score, generic_error
            else:
                reference, reference_validation, generic_score, generic_error = references[reference_key]
            smaller_candidate = None
            for modes in sorted(search.identity_candidates):
                index = len(results["candidates"])
                record = {"index": index, "identity_modes": modes, "policy": candidate_policy.record(),
                          "generic_score": generic_score, "generic_error": generic_error, "status": "running"}
                results["candidates"].append(record)
                try:
                    starts = [("generic", np.r_[np.zeros(modes), reference["per_view"].ravel()]
                               if reference is not None else None)]
                    if smaller_candidate is not None:
                        starts.append(("lower_capacity", np.r_[
                            smaller_candidate["identity"],
                            np.zeros(modes-len(smaller_candidate["identity"])),
                            smaller_candidate["per_view"].ravel()]))
                    fitted_starts = []
                    record["training_initializations"] = []
                    for source, initial in starts:
                        initialization = {"source": source}
                        record["training_initializations"].append(initialization)
                        try:
                            fitted_start = fit(model, train, candidate_policy, modes, initial=initial)
                            objective = float(fitted_start["report"]["objective"])
                            if not math.isfinite(objective):
                                raise ValueError("Nonfinite training objective")
                            initialization.update(status="completed", objective=objective)
                            fitted_starts.append((objective, source, fitted_start))
                        except (ValueError, RuntimeError) as error:
                            initialization.update(status="failed", error={
                                "type": type(error).__name__, "message": str(error)})
                    if not fitted_starts:
                        raise ValueError("Every training initialization failed")
                    _, source, candidate = min(fitted_starts, key=lambda item: item[0])
                    record["selected_training_initialization"] = source
                    smaller_candidate = candidate
                    tested = fit(model, validation, candidate_policy, modes,
                                 fixed_identity=candidate["identity"],
                                 initial=(reference_validation["per_view"].ravel()
                                          if reference_validation is not None else None))
                    score = residual_score(tested, validation)
                    record.update(status="completed", score=score,
                                  training=candidate["report"], heldout=tested["report"])
                    retain_candidate(record, candidate)
                    if best is None or score["view_balanced_rms"] < best[0]:
                        best = (score["view_balanced_rms"], model, candidate_policy,
                                candidate, tested, record)
                except (ValueError, RuntimeError) as error:
                    record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                    retain_candidate(record, None)
    if best is None:
        raise ValueError("Every identity search candidate failed; inspect retained candidate reports")
    score, model, selected_policy, candidate, tested, selected = best
    results["selected_index"] = selected["index"]
    results["selected_score"] = score
    results["matched_identity_comparison"] = [
        {"index": record["index"], "identity_modes": record["identity_modes"],
         "training_score": residual_score({"report": record["training"]}, train),
         "heldout_score": record["score"]}
        for record in results["candidates"]
        if record["status"] == "completed" and record["policy"] == selected["policy"]
    ]
    results["selection_limitations"] = (
        "Lowest observed landmark error, not proof of optimal capacity or likeness; "
        "training and held-out comparisons do not establish why a local optimizer chose this solution."
    )
    results["failed_candidates"] = sum(record["status"] == "failed" for record in results["candidates"])
    results["status"] = "completed_with_failures" if results["failed_candidates"] else "completed"
    per_view = np.empty((len(views), candidate["per_view"].shape[1]))
    per_view[training], per_view[heldout] = candidate["per_view"], tested["per_view"]
    final = fit(model, views, selected_policy, len(candidate["identity"]),
                initial=np.r_[candidate["identity"], per_view.ravel()])
    return model, final, selected_policy
