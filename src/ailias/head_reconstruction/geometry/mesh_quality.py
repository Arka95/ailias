"""Assess predicted mesh deformation without loading models or native editors."""

import numpy as np


def surface_quality(reference, candidate, triangles):
    """Reject collapsed/flipped predicted faces and excessive triangle-area changes.
    Compare existing topology only; passing does not certify likeness or absence of self-intersections.
    """
    before, after = (np.asarray(value)[triangles] for value in (reference, candidate))
    normals = [np.cross(value[:, 1]-value[:, 0], value[:, 2]-value[:, 0])
               for value in (before, after)]
    areas = [np.linalg.norm(value, axis=1) for value in normals]
    active = areas[0] > 1e-8
    ratio = areas[1][active]/areas[0][active]
    if not len(ratio) or not np.isfinite(ratio).all():
        raise ValueError("Prediction has no valid comparable surface")
    flipped = int(np.sum(np.einsum("ij,ij->i", *normals)[active] <= 0))
    quantiles = np.quantile(ratio, [.01, .5, .99])
    return {"flipped_or_collapsed_triangles": flipped, "area_ratio_quantiles": quantiles.tolist(),
            "passed": flipped == 0 and bool(quantiles[0] > .1 and quantiles[-1] < 10),
            "self_intersections_checked": False}
