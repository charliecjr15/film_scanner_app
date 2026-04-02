from __future__ import annotations
import numpy as np


def adjust_exposure(image: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    factor = 2.0 ** exposure
    return np.clip(image * factor, 0.0, 1.0)


def normalize_exposure_midtone(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    valid = gray[mask] if mask is not None and mask.shape == gray.shape and np.any(mask) else gray.reshape(-1)
    if valid.size < 100:
        return image

    mid = np.percentile(valid, 50)
    gain = 0.5 / (mid + 1e-6)
    gain = np.clip(gain, 0.5, 2.5)

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


def soft_highlight_rolloff(image: np.ndarray, strength: float = 0.15) -> np.ndarray:
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
            out[:, :, c][shadow_mask] * 0.7 +
            gray[shadow_mask] * 0.3
        )
        out[:, :, c][highlight_mask] = (
            out[:, :, c][highlight_mask] * 0.7 +
            gray[highlight_mask] * 0.3
        )

    return np.clip(out, 0.0, 1.0)


def render_border_soft(image: np.ndarray, border_mask: np.ndarray | None) -> np.ndarray:
    """
    Render border/rebate more gently so it doesn't compete with the image area.
    """
    if border_mask is None or not np.any(border_mask):
        return image

    out = image.copy()

    # Mild desaturation and softer highlight compression only on border area
    gray = np.mean(out, axis=2, keepdims=True)
    softened = gray + (out - gray) * 0.55
    softened = soft_highlight_rolloff(softened, 0.28)

    out[border_mask] = softened[border_mask]
    return np.clip(out, 0.0, 1.0)


def apply_filmic_contrast(image: np.ndarray) -> np.ndarray:
    image = np.power(np.clip(image, 0.0, 1.0), 0.95)
    image = image * image * (3.0 - 2.0 * image)
    return np.clip(image, 0.0, 1.0)