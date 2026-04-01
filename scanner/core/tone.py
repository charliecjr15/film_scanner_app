from __future__ import annotations
import numpy as np

def adjust_exposure(image: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    factor = 2.0 ** exposure
    return np.clip(image * factor, 0.0, 1.0)

def apply_levels(image: np.ndarray, black_point: float = 0.0, white_point: float = 1.0) -> np.ndarray:
    white_point = max(white_point, black_point + 1e-5)
    out = (image - black_point) / (white_point - black_point)
    return np.clip(out, 0.0, 1.0)

def adjust_contrast(image: np.ndarray, contrast: float = 0.0) -> np.ndarray:
    factor = 1.0 + contrast
    midpoint = 0.5
    out = (image - midpoint) * factor + midpoint
    return np.clip(out, 0.0, 1.0)

def soft_highlight_rolloff(image: np.ndarray, strength: float = 0.15) -> np.ndarray:
    if strength <= 0:
        return image

    x = np.clip(image, 0.0, 1.0)
    mask = np.clip((x - 0.7) / 0.3, 0.0, 1.0)
    compressed = 0.7 + (x - 0.7) * (1.0 - strength * mask)
    out = np.where(x > 0.7, compressed, x)
    return np.clip(out, 0.0, 1.0)