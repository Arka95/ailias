"""Typed evidence, geometry, fitting results and diagnostics for head reconstruction."""

from dataclasses import dataclass, field, fields
import json
from typing import Literal

import numpy as np

from ailias.common.json_values import to_json_value
from ailias.head_reconstruction.settings import FitPolicy


class _JsonRecord:
    """Serialize reconstruction records without exposing NumPy objects or writing artifacts.
    Keep array conversion local to this numerical domain rather than changing generic JSON handling.
    """

    def to_dict(self) -> dict:
        """Return strict JSON-compatible fields, including nested records and numerical arrays.
        Reject nonfinite or unsupported values instead of silently stringifying runtime objects.
        """
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return to_json_value(
            {
                name: value.tolist()
                if isinstance(value, (np.ndarray, np.generic))
                else value
                for name, value in values.items()
            }
        )

    def __str__(self) -> str:
        """Serialize this record as strict JSON entirely in memory.
        Leave all file creation and output-directory decisions to the caller.
        """
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class HeadViewObservation(_JsonRecord):
    """Retain one view's validated pixel landmarks or its explicit evidence failure.
    Foreground support is not independent visibility, and no depth measurement is implied.
    """

    view_id: str
    status: Literal["ready", "review"]
    source_size: tuple[int, int] | None = None
    source_xy: np.ndarray | None = field(default=None, repr=False)
    on_head: np.ndarray | None = field(default=None, repr=False)
    # Outer eye-corner distance in source pixels, not physical interpupillary distance.
    eye_span_source: float | None = None
    reason: str | None = None
    landmark_depth_is_measured: bool = False
    per_point_visibility_available: bool = False


@dataclass(frozen=True)
class ExcludedHeadView(_JsonRecord):
    """Associate a numerical/evidence exclusion with a view, not a person identity.
    Preserve the reason even when too few remaining views allow fitting to start.
    """

    view_id: str
    reason: str


@dataclass
class ReconstructionInputs(_JsonRecord):
    """Track usable input evidence and the actual training/validation view association.
    Counts describe this reconstruction, not imageprocessing's collection-selection decisions.
    """

    observation_count: int
    landmarks_ready: int = 0
    training_view_ids: list[str] = field(default_factory=list)
    validation_view_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FitEvaluation(_JsonRecord):
    """Retain category-balanced held-out error separately from the median across all views.
    Errors are normalized by predicted outer-eye span, not a calibrated physical distance.
    """

    view_balanced_mean_error: float
    per_initial_view_category: dict[str, float]
    median_view_error: float | None = None


@dataclass(frozen=True)
class IdentityCandidateEvaluation(_JsonRecord):
    """Describe one identity-capacity candidate and whether frontal accuracy was preserved.
    Keep evaluation separate from the fitted numerical parameter arrays.
    """

    identity_modes: int
    score: FitEvaluation
    frontal_preserved: bool


@dataclass(frozen=True)
class FittingRuntime(_JsonRecord):
    """Retain common solver execution measurements and backend-specific scalar metadata.
    Optional backend details preserve CUDA provenance without importing or probing that runtime.
    """

    backend: str
    device: str
    seconds: float
    evaluations: int
    converged: bool
    termination: str
    objective: float
    backend_details: dict[str, str | int | float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class FittedHeadView(_JsonRecord):
    """Keep per-view reprojection measurements associated with their original view identifier.
    Orientation remains an uncalibrated numerical-camera diagnostic, not an anatomical certification.
    """

    view_id: str
    supported_points: int
    median_outer_eye_span_error: float
    p90_outer_eye_span_error: float
    weighted_rms_outer_eye_span_error: float
    median_source_pixel_error: float
    focal_factor: float
    orientation: dict


@dataclass(frozen=True)
class HeadFitSummary(_JsonRecord):
    """Separate measured fitting diagnostics from the numerical parameters that produced them.
    Report per-view errors and runtime without treating reprojection as visual likeness approval.
    """

    identity_modes: int
    expression_modes: int
    views: list[FittedHeadView]
    runtime: FittingRuntime
    median_outer_eye_span_error: float


@dataclass(frozen=True)
class HeadFitParameters(_JsonRecord):
    """Retain shared identity, per-view camera/expression and optimizer arrays without a native scene.
    Row order follows the fitted view summaries; projected landmarks use normalized fitting coordinates.
    """

    identity_coefficients: np.ndarray = field(repr=False)
    cameras_and_expressions: np.ndarray = field(repr=False)
    projected_landmarks: np.ndarray = field(repr=False)
    optimizer_parameters: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class HeadFittingResult(_JsonRecord):
    """Associate fitted numerical parameters with their structured execution/error summary.
    Keep this independent of mesh construction, artifact export and native-engine lifetime.
    """

    parameters: HeadFitParameters
    summary: HeadFitSummary


@dataclass(frozen=True)
class HeadMeshQuality(_JsonRecord):
    """Describe template-relative triangle deformation without claiming complete mesh validity.
    Surface checks do not certify likeness or absence of self-intersections.
    """

    flipped_or_collapsed_triangles: int
    area_ratio_quantiles: list[float]
    passed: bool
    self_intersections_checked: bool = False


@dataclass(frozen=True)
class HeadMesh(_JsonRecord):
    """Retain neutral vertex positions and matching triangle topology without camera or expression.
    Coordinates remain in the supplied prior's uncalibrated model space.
    """

    vertices: np.ndarray = field(repr=False)
    triangles: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class ReconstructionFailure(_JsonRecord):
    """Keep the original failure category and explanation alongside completed diagnostics.
    Failure is not a successful reconstruction using a substituted template.
    """

    error_type: str
    message: str


@dataclass
class ReconstructionReport(_JsonRecord):
    """Accumulate stage-specific evidence, candidate evaluations and explicit reconstruction outcome.
    Leave mesh arrays and detailed fitted-view metrics in their separate result records.
    """

    policy: FitPolicy
    inputs: ReconstructionInputs
    status: Literal["fitting", "failed", "awaiting_human_likeness_review"] = "fitting"
    prepared_view_ids: list[str] = field(default_factory=list)
    excluded: list[ExcludedHeadView] = field(default_factory=list)
    candidates: list[IdentityCandidateEvaluation] = field(default_factory=list)
    baseline_score: FitEvaluation | None = None
    heldout_improvement: float | None = None
    mesh_quality: HeadMeshQuality | None = None
    failure: ReconstructionFailure | None = None
    limitations: tuple[str, ...] = (
        "Predicted landmarks constrain a supplied head prior, not a measured 3D scan.",
        "Scale, focal length and hidden anatomy remain uncalibrated or prior-derived.",
        "Alpha and template normals are support heuristics, not per-point visibility ground truth.",
        "Photo colours, orientation estimates and MediaPipe blendshapes are not geometry coefficients.",
        "Reprojection and surface checks do not certify visual likeness or self-intersection freedom.",
    )


@dataclass(frozen=True)
class HeadReconstructionResult(_JsonRecord):
    """Return one neutral head with typed observations, fitting parameters and reconstruction diagnostics.
    Support explicit JSON export without retaining source images or automatically writing files.
    """

    mesh: HeadMesh
    fit: HeadFittingResult
    observations: list[HeadViewObservation]
    report: ReconstructionReport
