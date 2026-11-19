"""Explicit photo-projection inputs, independent of upstream workflow manifests."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PhotoProjectionInput:
    """Describe one unchanged fitting grid and the measured appearance available for projection.
    Paths are resolved/verified by the caller; masks are existing evidence, not rerun inference.
    """

    image: Path
    regions: Path
    head_image: Path
    source_size: tuple[int, int]
    skin_linear_rgb: list[float] | None = None
    blendshapes: dict[str, float] = field(default_factory=dict)
