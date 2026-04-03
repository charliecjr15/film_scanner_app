from __future__ import annotations

import numpy as np


def adjust_exposure(image: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    factor = 2.0 ** exposure
    return np.clip(image * factor, 0.0, 1.0)


def normalize_exposure_midtone(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    gray = np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])

    valid = gray[scene_mask] if scene_mask is not None and scene_mask.shape == gray.shape and np.any(scene_mask) else gray.reshape(-1)
    if valid.size < 100:
        return image

    mid = float(np.percentile(valid, 50.0))
    gain = 0.56 / max(mid, 1e-6)
    gain = np.clip(gain, 0.85, 1.45)

    return np.clip(image * gain, 0.0, 1.0)


def apply_levels(image: np.ndarray, black_point: float = 0.0, white_point: float = 1.0) -> np.ndarray:
    white_point = max(white_point, black_point + 1e-5)
    out = (image - black_point) / (white_point - black_point)
    return np.clip(out, 0.0, 1.0)


def adjust_contrast(image: np.ndarray, contrast: float = 0.0) -> np.ndarray:
    factor = 1.0 + contrast
    midpoint = 0.5
    out = (image - midpoint) * factor + midpoint
    return np.clip(out, 0.0, 1.0)


def soft_highlight_rolloff(image: np.ndarray, strength: float = 0.10) -> np.ndarray:
    if strength <= 0:
        return image

    x = np.clip(image, 0.0, 1.0)
    mask = np.clip((x - 0.72) / 0.28, 0.0, 1.0)
    compressed = 0.72 + (x - 0.72) * (1.0 - strength * mask)
    out = np.where(x > 0.72, compressed, x)
    return np.clip(out, 0.0, 1.0)


def protect_extremes(image: np.ndarray) -> np.ndarray:
    gray = np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])

    shadow_mask = gray < 0.11
    highlight_mask = gray > 0.92

    out = image.copy()

    for c in range(3):
        out[:, :, c][shadow_mask] = (
            out[:, :, c][shadow_mask] * 0.84 +
            gray[shadow_mask] * 0.16
        )
        out[:, :, c][highlight_mask] = (
            out[:, :, c][highlight_mask] * 0.84 +
            gray[highlight_mask] * 0.16
        )

    return np.clip(out, 0.0, 1.0)


def apply_filmic_contrast(image: np.ndarray) -> np.ndarray:
    """
    Gentle S-curve:
    - opens lower mids
    - protects highlights
    - avoids the dead/muddy look from linear rendering
    """
    x = np.clip(image, 0.0, 1.0)

    # slight toe lift
    toe = np.power(x, 0.92)

    # soft S-curve centered around mids
    out = toe * toe * (3.0 - 2.0 * toe)

    # blend a bit of the original back to avoid plastic look
    out = out * 0.82 + x * 0.18
    return np.clip(out, 0.0, 1.0)
