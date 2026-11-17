"""Translate measured photo/render differences into bounded native edit proposals."""

import numpy as np


def iris_tint_adjustment(reference, rendered, current_tint):
    """Propose bounded multipliers; only a subsequent native render can validate them."""
    target = np.asarray(reference["median_linear_rgb"], dtype=float)
    actual = np.asarray(rendered["median_linear_rgb"], dtype=float)
    current = np.asarray(current_tint, dtype=float)
    if (any(value.shape != (3,) for value in (target, actual, current))
            or not all(np.isfinite(value).all() for value in (target, actual, current))
            or np.any(target < 0) or np.any(target > 1)
            or np.any(actual <= 1e-8) or np.any(actual > 1)
            or np.any(current <= 0) or np.any(current > 1)):
        raise ValueError("Tint adjustment requires finite usable linear RGB evidence")
    proposed = current*target/actual
    return {
        "eyes": {"global_tint_linear": np.clip(proposed, 0, 1).tolist()},
        "clipped_channels": np.flatnonzero(proposed > 1).tolist(),
        "reference": reference, "rendered": rendered, "current_tint_linear": current.tolist(),
        "render_verification_required": True, "intrinsic_colour_calibrated": False,
    }
