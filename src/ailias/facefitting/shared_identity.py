"""Fit one identity against held-out views, keeping every view's expression separate."""

import numpy as np

from ailias.facefitting.fitting import evaluation_score, fit, validation_indices


def fit_shared_identity(model, views, policy, report):
    heldout = validation_indices(views)
    training = [index for index in range(len(views)) if index not in heldout]
    train, validation = ([views[index] for index in indices] for indices in (training, heldout))
    report["inputs"].update(training=[view["face_id"] for view in train],
                            heldout=[view["face_id"] for view in validation])
    baseline_train = fit(model, train, policy, 0, fixed_identity=np.zeros(0))
    baseline_validation = fit(model, validation, policy, 0, fixed_identity=np.zeros(0))
    baseline_score = evaluation_score(baseline_validation["report"], validation)
    candidates, best = [], None
    for modes in policy.identity_candidates:
        candidate = fit(model, train, policy, modes, initial=np.r_[
            np.zeros(modes), baseline_train["per_view"].ravel()])
        tested = fit(model, validation, policy, modes, fixed_identity=candidate["identity"],
                     initial=baseline_validation["per_view"].ravel())
        score = evaluation_score(tested["report"], validation)
        front = score["per_initial_view_category"].get("front")
        base_front = baseline_score["per_initial_view_category"].get("front")
        preserved = front is None or base_front is None or front <= base_front*1.05
        candidates.append({"modes": modes, "score": score, "frontal_preserved": preserved})
        if preserved and (best is None or score["view_balanced_mean_error"] < best[0]):
            best = score["view_balanced_mean_error"], candidate, tested
    report["candidates"] = candidates
    report["baseline_score"] = baseline_score
    if best is None or best[0] >= baseline_score["view_balanced_mean_error"]*.99:
        raise ValueError("Existing shared-identity fit criterion failed; no personalized head published")
    score, candidate, tested = best
    report["heldout_improvement"] = 1-score/baseline_score["view_balanced_mean_error"]
    per_view = np.empty((len(views), candidate["per_view"].shape[1]))
    per_view[training], per_view[heldout] = candidate["per_view"], tested["per_view"]
    return fit(model, views, policy, len(candidate["identity"]),
               initial=np.r_[candidate["identity"], per_view.ravel()])
