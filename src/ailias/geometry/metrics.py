"""Align corresponding landmarks and measure landmark or sampled-surface errors.

Distinguishes metric units from scale-invariant fits and rejects reflected alignment.
"""

import math


def points(values, dimension, maximum=10000):
    import numpy as np

    if not isinstance(values, list) or not 1 <= len(values) <= maximum:
        raise ValueError(f"Expected 1-{maximum} points")
    if any(not isinstance(row, list) or len(row) != dimension
           or any(isinstance(v, bool) or not isinstance(v, (int, float))
                  or not math.isfinite(v) for v in row) for row in values):
        raise ValueError("Points must have consistent dimensions and finite numeric coordinates")
    return np.asarray(values, dtype=np.float64)


def statistics(distances):
    import numpy as np

    return {"mean": float(np.mean(distances)), "median": float(np.median(distances)),
            "rmse": float(np.sqrt(np.mean(np.square(distances)))),
            "p95": float(np.percentile(distances, 95)), "max": float(np.max(distances))}


def align(reference, candidate, fit="rigid"):
    import numpy as np

    if fit not in {"rigid", "similarity"}:
        raise ValueError("fit must be rigid or similarity")
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Alignment requires corresponding equal-sized point arrays")
    dimension = reference.shape[1]
    if dimension not in {2, 3} or len(reference) < dimension + 1:
        raise ValueError("Use at least three 2D or four 3D corresponding landmarks")
    reference_mean, candidate_mean = reference.mean(0), candidate.mean(0)
    target, moving = reference - reference_mean, candidate - candidate_mean
    if min(np.linalg.matrix_rank(target), np.linalg.matrix_rank(moving)) < dimension - 1:
        raise ValueError("Degenerate landmark geometry cannot define alignment")
    left, singular, right = np.linalg.svd(moving.T @ target)
    signs = np.ones(dimension)
    signs[-1] = 1 if np.linalg.det(left @ right) >= 0 else -1
    rotation = left @ np.diag(signs) @ right
    scale = float(np.sum(singular * signs) / np.sum(moving * moving)) if fit == "similarity" else 1.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Alignment produced a non-positive scale")
    translation = reference_mean - scale * candidate_mean @ rotation
    transform = {"scale": scale, "rotation": rotation.tolist(),
                 "translation": translation.tolist(), "reflection_allowed": False,
                 "convention": "aligned_candidate = scale * candidate @ rotation + translation"}
    return scale * candidate @ rotation + translation, transform


def nearest_distances(first, second):
    import numpy as np

    distances = []
    for offset in range(0, len(first), 128):
        difference = first[offset:offset + 128, None, :] - second[None, :, :]
        squared = np.einsum("ijk,ijk->ij", difference, difference)
        distances.extend(np.sqrt(squared.min(axis=1)).tolist())
    return np.asarray(distances)


def compare_geometry(reference: dict, candidate: dict, fit="rigid") -> dict:
    import numpy as np

    dimension = reference.get("dimension")
    if isinstance(dimension, bool) or dimension not in {2, 3} or candidate.get("dimension") != dimension:
        raise ValueError("Both inputs must have the same dimension, 2 or 3")
    if dimension == 3:
        factors = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
        if reference.get("units") not in factors or candidate.get("units") not in factors:
            raise ValueError("3D geometry needs explicit mm, cm or m units")
        ref_factor, candidate_factor = factors[reference["units"]], factors[candidate["units"]]
        units = "mm"
    else:
        if reference.get("units") not in {"pixels", "normalized"} or reference.get("units") != candidate.get("units"):
            raise ValueError("2D geometry needs matching pixels or normalized coordinate units")
        ref_factor = candidate_factor = 1.0
        units = reference["units"]
    first, second = reference.get("landmarks"), candidate.get("landmarks")
    if (not isinstance(first, dict) or not isinstance(second, dict)
            or set(first) != set(second) or not {"left_eye", "right_eye"} <= set(first)):
        raise ValueError("Inputs need identical named landmark sets including left_eye and right_eye")
    names = sorted(first)
    target = points([first[name] for name in names], dimension) * ref_factor
    moving = points([second[name] for name in names], dimension) * candidate_factor
    aligned, transform = align(target, moving, fit)
    interocular = float(np.linalg.norm(target[names.index("left_eye")] - target[names.index("right_eye")]))
    if interocular <= 1e-12:
        raise ValueError("Reference eye landmarks must be distinct")
    errors = np.linalg.norm(target - aligned, axis=1)
    result = {
        "dimension": dimension, "units": units, "fit": fit,
        "landmark_count": len(names), "landmark_error": statistics(errors),
        "interocular_distance": interocular,
        "normalized_rmse": float(np.sqrt(np.mean(errors ** 2)) / interocular),
        "normalization": "reference interocular distance; lower is better, not a likeness percentage",
        "per_landmark": {name: float(error) for name, error in zip(names, errors)},
        "candidate_to_reference": transform,
        "is_3d_geometry": dimension == 3,
        "limitations": [
            "Requires correct landmark correspondences and compatible neutral expressions",
            "2D error is viewpoint/camera dependent and cannot establish 3D shape likeness",
            "Similarity alignment removes absolute size differences; rigid fitting preserves scale",
            "No calibrated pass/fail threshold or facial-likeness percentage",
        ],
    }
    regions = reference.get("regions", {})
    if not isinstance(regions, dict):
        raise ValueError("regions must map region names to landmark names")
    result["regions"] = {}
    for region, members in regions.items():
        if (not isinstance(members, list) or not members or len(set(members)) != len(members)
                or not set(members) <= set(names)):
            raise ValueError("Each region needs distinct known landmark names")
        result["regions"][region] = statistics(errors[[names.index(name) for name in members]])
    if "surface_points" in reference or "surface_points" in candidate:
        if dimension != 3 or "surface_points" not in reference or "surface_points" not in candidate:
            raise ValueError("Both 3D inputs must provide surface_points for sampled-surface metrics")
        ref_surface = points(reference["surface_points"], 3) * ref_factor
        candidate_surface = points(candidate["surface_points"], 3) * candidate_factor
        rotation = np.asarray(transform["rotation"])
        candidate_surface = transform["scale"] * candidate_surface @ rotation + transform["translation"]
        forward = nearest_distances(ref_surface, candidate_surface)
        backward = nearest_distances(candidate_surface, ref_surface)
        result["sampled_surface"] = {
            "reference_points": len(ref_surface), "candidate_points": len(candidate_surface),
            "reference_to_candidate": statistics(forward),
            "candidate_to_reference": statistics(backward),
            "symmetric_chamfer_mean": float((forward.mean() + backward.mean()) / 2),
            "hausdorff_max": float(max(forward.max(), backward.max())),
            "hausdorff_p95": float(max(np.percentile(forward, 95), np.percentile(backward, 95))),
            "definition": "Unsquared Euclidean nearest-sample distances after landmark alignment",
            "limitations": "Not point-to-triangle distance or NoW evaluation; sampling, crop and density matter",
        }
    return result
