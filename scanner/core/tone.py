from __future__ import annotations
import numpy as np


def adjust_exposure(image: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    factor = 2.0 ** exposure
    return np.clip(image * factor, 0.0, 1.0)


def normalize_exposure_midtone(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    valid = gray[scene_mask] if scene_mask is not None and scene_mask.shape == gray.shape and np.any(scene_mask) else gray.reshape(-1)
    if valid.size < 100:
        return image

    mid = np.percentile(valid, 50)
    gain = 0.50 / (mid + 1e-6)
    gain = np.clip(gain, 0.75, 1.60)

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
    mask = np.clip((x - 0.7) / 0.3, 0.0, 1.0)
    compressed = 0.7 + (x - 0.7) * (1.0 - strength * mask)
    out = np.where(x > 0.7, compressed, x)
    return np.clip(out, 0.0, 1.0)


def protect_extremes(image: np.ndarray) -> np.ndarray:
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    shadow_mask = gray < 0.1
    highlight_mask = gray > 0.9

    out = image.copy()

    for c in range(3):
        out[:, :, c][shadow_mask] = (
            out[:, :, c][shadow_mask] * 0.82 +
            gray[shadow_mask] * 0.18
        )
        out[:, :, c][highlight_mask] = (
            out[:, :, c][highlight_mask] * 0.82 +
            gray[highlight_mask] * 0.18
        )

    return np.clip(out, 0.0, 1.0)


def suppress_outer_area(
    image: np.ndarray,
    border_mask: np.ndarray | None = None,
) -> np.ndarray:
    if border_mask is None or not np.any(border_mask):
        return image

    out = image.copy()
    gray = np.mean(out, axis=2, keepdims=True)
    subdued = gray + (out - gray) * 0.20
    subdued = soft_highlight_rolloff(subdued, 0.45)
    out[border_mask] = subdued[border_mask]
    return np.clip(out, 0.0, 1.0)


def apply_filmic_contrast(image: np.ndarray) -> np.ndarray:
    image = np.power(np.clip(image, 0.0, 1.0), 0.98)
    image = image * image * (3.0 - 2.0 * image)
    return np.clip(image, 0.0, 1.0)