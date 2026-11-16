import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.json"


@dataclass(frozen=True)
class Policy:
    min_photos: int = 8
    max_photos: int = 15
    min_short_side: int = 1024
    min_edge_energy: float = 18.0
    max_dark_fraction: float = 0.6
    max_bright_fraction: float = 0.6
    angle_tolerance: float = 10.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for name in ("min_photos", "max_photos", "min_short_side"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.min_photos <= self.max_photos or self.min_short_side < 1:
            raise ValueError("Invalid photo count or resolution limits")
        if self.min_edge_energy < 0:
            raise ValueError("min_edge_energy cannot be negative")
        if not 0 < self.angle_tolerance < 15:
            raise ValueError("angle_tolerance must be between 0 and 15 degrees")
        for value in (self.max_dark_fraction, self.max_bright_fraction):
            if not 0 <= value <= 1:
                raise ValueError("Exposure fractions must be between 0 and 1")


def load_settings(path: Path = DEFAULT_CONFIG) -> tuple[Policy, Path]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    root = config.pop("data_root", r"..\data")
    if not isinstance(root, str) or not root:
        raise ValueError("data_root must be a non-empty path")
    policy = Policy(**config)
    data_root = Path(root)
    if not data_root.is_absolute():
        data_root = path.resolve().parent / data_root
    return policy, data_root.resolve()
