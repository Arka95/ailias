"""Synthetic solver/result records shared by reconstruction consumer unit tests."""

import numpy as np

from ailias.head_reconstruction.geometry.fitting_results import (
    fitting_result_from_solver,
)


def make_solver_fit(view_ids, *, identity=None, per_view=None, error=0.1):
    """Build finite synthetic backend output without invoking an optimizer or a model.
    Keep actual view association and parameter arrays available to the tests.
    """
    identity = np.zeros(1) if identity is None else identity
    per_view = np.zeros((len(view_ids), 8)) if per_view is None else per_view
    return {
        "identity": identity,
        "per_view": per_view,
        "parameters": np.r_[identity, per_view.ravel()],
        "projected": np.zeros((len(view_ids), 468, 2)),
        "report": {
            "identity_modes": len(identity),
            "expression_modes": per_view.shape[1] - 7,
            "backend": "synthetic",
            "device": "cpu",
            "seconds": 0.0,
            "nfev": 1,
            "converged": True,
            "termination": "fixture",
            "objective": error,
            "median_outer_eye_span_error": error,
            "views": [
                {
                    "face_id": view_id,
                    "supported_points": 120,
                    "median_outer_eye_span_error": error,
                    "p90_outer_eye_span_error": error,
                    "weighted_rms_outer_eye_span_error": error,
                    "median_source_pixel_error": error,
                    "focal_factor": 1.0,
                    "view": {"category": "front"},
                }
                for view_id in view_ids
            ],
        },
    }


def make_head_fit(view_ids, **kwargs):
    """Build the real typed fitting contract from synthetic numerical output.
    Use production conversion so consumer fixtures cannot silently retain an obsolete dictionary API.
    """
    return fitting_result_from_solver(make_solver_fit(view_ids, **kwargs))
