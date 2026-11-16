import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

from ailias.settings import PROJECT_ROOT

DEFAULT_GENERATION_CONFIG = PROJECT_ROOT / "config" / "generation.json"
TOOL_SPECS = PROJECT_ROOT / "config" / "generation-tools.json"


@dataclass(frozen=True)
class GenerationPolicy:
    max_images: int = 32
    min_registered_images: int = 5
    min_registered_fraction: float = 0.6
    min_sparse_points: int = 100
    max_reprojection_error_px: float = 2.5
    mapper_init_min_inliers: int = 100
    mapper_init_min_tri_angle: float = 16.0
    mapper_abs_pose_min_inliers: int = 30
    stereo_min_consistent: int = 2
    fusion_min_pixels: int = 5
    max_feature_image_size: int = 2000
    max_dense_image_size: int = 1200
    min_fused_points: int = 1000
    min_mesh_faces: int = 500
    cpu_threads: int = 4
    gpu_index: int = 0
    timeout_seconds: int = 1200
    poisson_depth: int = 9
    poisson_trim: float = 5.0
    render_size: int = 768
    render_samples: int = 16
    blender_device: str = "CUDA"

    def __post_init__(self):
        floats = {"min_registered_fraction", "max_reprojection_error_px", "poisson_trim",
                  "mapper_init_min_tri_angle"}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "blender_device":
                if value not in {"CUDA", "OPTIX", "CPU"}:
                    raise ValueError("blender_device must be CUDA, OPTIX or explicitly CPU")
                continue
            minimum = 0 if field.name == "gpu_index" else 1
            if field.name in floats:
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"{field.name} must be finite and positive")
            elif isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field.name} must be an integer >= {minimum}")
        if not 0 < self.min_registered_fraction <= 1:
            raise ValueError("min_registered_fraction must be at most 1")
        if not 3 <= self.min_registered_images <= self.max_images <= 64:
            raise ValueError("Require 3 <= min_registered_images <= max_images <= 64")
        if not 5 <= self.poisson_depth <= 12:
            raise ValueError("poisson_depth must be between 5 and 12")
        if self.mapper_init_min_tri_angle >= 180:
            raise ValueError("mapper_init_min_tri_angle must be less than 180 degrees")


def load_settings(path: Path = DEFAULT_GENERATION_CONFIG):
    values = json.loads(path.read_text(encoding="utf-8"))
    if set(values) != {"tools_root", "policy"}:
        raise ValueError("Generation configuration requires tools_root and policy only")
    root = Path(values["tools_root"])
    root = root.resolve() if root.is_absolute() else (path.resolve().parent / root).resolve()
    return GenerationPolicy(**values["policy"]), root
