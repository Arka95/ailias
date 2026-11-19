"""Validate portrait reel policy and compute lens framing independently of Unreal."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ailias.settings import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "reanimation.json"


@dataclass(frozen=True)
class ReelPolicy:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    focal_length_mm: float = 85
    sensor_height_mm: float = 36
    framing_height_cm: float = 52
    head_center_offset_cm: float = 6
    front_axis: str = "+Y"
    character_location_cm: tuple = (0, 0, 0)
    character_yaw_degrees: float = 0
    subject_screen_position: tuple = (0.5, 0.5)
    background_srgb: tuple = (0, 1, 0)
    exposure_compensation_ev: float = -5.5
    mood_intensity: float = 0.7
    key_lumens: float = 900
    fill_ratio: float = 0.65
    light_size_cm: float = 200
    temperature_kelvin: float = 5600
    temporal_samples: int = 1
    warmup_frames: int = 8
    encoder: str = "h264_nvenc"
    max_audio_seconds: float = 120
    timeout_seconds: int = 1800

    def __post_init__(self):
        for name in (
            "width",
            "height",
            "fps",
            "temporal_samples",
            "warmup_frames",
            "timeout_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "focal_length_mm",
            "sensor_height_mm",
            "framing_height_cm",
            "head_center_offset_cm",
            "mood_intensity",
            "key_lumens",
            "fill_ratio",
            "temperature_kelvin",
            "max_audio_seconds",
            "exposure_compensation_ev",
            "light_size_cm",
            "character_yaw_degrees",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.width * 16 != self.height * 9 or self.width % 2 or self.height % 2:
            raise ValueError("Require even 9:16 output dimensions")
        if not (1 <= self.fps <= 60 and self.width <= 2160 and self.height <= 3840):
            raise ValueError("Bound output to at most 2160x3840 and 60 fps")
        if not (
            30 <= self.focal_length_mm <= 150
            and 10 <= self.sensor_height_mm <= 50
            and 25 <= self.framing_height_cm <= 100
        ):
            raise ValueError("Invalid close-up lens/framing dimensions")
        if not 0 <= self.mood_intensity <= 1 or not 0 < self.fill_ratio <= 1:
            raise ValueError("Mood and fill ratio must be in range")
        if not -12 <= self.exposure_compensation_ev <= 12:
            raise ValueError("Exposure compensation must be between -12 and 12 EV")
        if not 50 <= self.light_size_cm <= 400:
            raise ValueError("Soft light size must be between 50 and 400 cm")
        if not -180 <= self.character_yaw_degrees <= 180:
            raise ValueError("Character yaw must be between -180 and 180 degrees")
        for name, count in (
            ("character_location_cm", 3),
            ("subject_screen_position", 2),
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, (tuple, list))
                or len(values) != count
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in values
                )
            ):
                raise ValueError(f"{name} requires {count} finite coordinates")
        if any(not 0 <= value <= 1 for value in self.subject_screen_position):
            raise ValueError("Screen position must use normalized coordinates in [0,1]")
        if not (
            0 < self.key_lumens <= 10000
            and 3000 <= self.temperature_kelvin <= 8000
            and 0 < self.max_audio_seconds <= 600
        ):
            raise ValueError("Invalid lighting or bounded audio duration")
        if len(self.background_srgb) != 3 or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in self.background_srgb
        ):
            raise ValueError("Background requires three finite sRGB values in [0,1]")
        if self.front_axis not in {"+X", "-X", "+Y", "-Y"}:
            raise ValueError(
                "front_axis must declare the rig's neutral facing direction"
            )
        if self.encoder not in {"h264_nvenc", "libx264"}:
            raise ValueError(
                "Use explicit NVIDIA NVENC or explicit CPU libx264; no silent fallback"
            )

    def framing(self):
        return {
            "sensor_width_mm": self.sensor_height_mm * self.width / self.height,
            "sensor_height_mm": self.sensor_height_mm,
            "camera_distance_cm": self.framing_height_cm
            * self.focal_length_mm
            / self.sensor_height_mm,
            "horizontal_fov_degrees": math.degrees(
                2
                * math.atan(
                    self.sensor_height_mm
                    * self.width
                    / self.height
                    / (2 * self.focal_length_mm)
                )
            ),
            "vertical_fov_degrees": math.degrees(
                2 * math.atan(self.sensor_height_mm / (2 * self.focal_length_mm))
            ),
            "visible_height_cm": self.framing_height_cm,
            "visible_width_cm": self.framing_height_cm * self.width / self.height,
        }

    def to_dict(self):
        return asdict(self)

    def placement(self):
        x, y = {"+X": (1, 0), "-X": (-1, 0), "+Y": (0, 1), "-Y": (0, -1)}[
            self.front_axis
        ]
        angle = math.radians(self.character_yaw_degrees)
        front = (
            x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle),
            0,
        )
        right = (front[1], -front[0], 0)
        u, v = self.subject_screen_position
        framing = self.framing()
        shift = -(u - 0.5) * framing["visible_width_cm"]
        return {
            "front": front,
            "right": right,
            "camera_target_shift_cm": (
                right[0] * shift,
                right[1] * shift,
                (v - 0.5) * framing["visible_height_cm"],
            ),
        }


def load_policy(path: Path = DEFAULT_CONFIG):
    return ReelPolicy(**json.loads(path.read_text(encoding="utf-8")))
