from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

NormalizedCropRect = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class ImageJob:
    source_path: str
    film_type: str = "color_negative"

    rotation: int = 0
    flip_h: bool = False
    flip_v: bool = False

    normalized_crop_rect: Optional[NormalizedCropRect] = None
    include_border: bool = False
    auto_crop_enabled: bool = True

    exposure: float = 0.0
    temp: float = 0.0
    tint: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    black_point: float = 0.0
    white_point: float = 1.0
    sharpness: float = 0.25

    gray_pick_normalized: Optional[Point] = None
    preset_name: str = "Balanced"
    output_icc_profile: str = "sRGB IEC61966-2.1"

    def display_name(self) -> str:
        return self.source_path.split("\\")[-1].split("/")[-1]