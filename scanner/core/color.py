from __future__ import annotations

import numpy as np


def _sample_pixels(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    if scene_mask is not None and scene_mask.shape[:2] == image.shape[:2] and np.any(scene_mask):
        sample = image[scene_mask]
        if sample.shape[0] >= 64:
            return sample
    return image.reshape(-1, 3)


def auto_balance(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Restrained gray-balance on midtones only.

    This should refine a good inversion, not rescue a bad one.
    """
    sample = _sample_pixels(image, scene_mask)
    if sample.shape[0] < 64:
        return image

    luma = sample @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mids = sample[(luma > 0.18) & (luma < 0.82)]
    if mids.shape[0] < 64:
        mids = sample

    means = np.mean(mids, axis=0).astype(np.float32)
    target = float(np.mean(means))
    gains = target / np.maximum(means, 1e-5)
    gains = np.clip(gains, 0.92, 1.08)

    luma_full = np.mean(image, axis=2, keepdims=True)
    mid_weight = np.clip((luma_full - 0.16) / 0.22, 0.0, 1.0) * (1.0 - np.clip((luma_full - 0.82) / 0.12, 0.0, 1.0))
    out = image * (1.0 + (gains.reshape(1, 1, 3) - 1.0) * mid_weight)
    return np.clip(out, 0.0, 1.0)


def apply_filmic_color_balance(
    image: np.ndarray,
    scene_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Mild photographic color rebuild after inversion.

    Goal:
    - preserve skin and warm midtones
    - keep shadows from drifting cyan/green
    - avoid aggressive auto-coloring
    """
    sample = _sample_pixels(image, scene_mask)
    if sample.shape[0] < 128:
        return image

    luma = sample @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mids = sample[(luma > 0.20) & (luma < 0.78)]
    if mids.shape[0] < 128:
        mids = sample

    means = np.mean(mids, axis=0).astype(np.float32)

    target = np.array(
        [
            means[1] * 1.02,
            means[1] * 1.00,
            means[1] * 0.97,
        ],
        dtype=np.float32,
    )
    gains = np.clip(target / np.maximum(means, 1e-5), 0.94, 1.08)

    luma_full = np.mean(image, axis=2, keepdims=True)
    shadow_to_mid = np.clip((luma_full - 0.14) / 0.28, 0.0, 1.0)
    mid_to_high = 1.0 - np.clip((luma_full - 0.78) / 0.14, 0.0, 1.0)
    weight = shadow_to_mid * mid_to_high

    out = image * (1.0 + (gains.reshape(1, 1, 3) - 1.0) * weight)

    # tiny warm push in mids; not enough to orange-stain highlights
    bias = np.concatenate(
        [
            weight * 0.010,
            np.zeros_like(weight),
            -weight * 0.010,
        ],
        axis=2,
    )
    return np.clip(out + bias, 0.0, 1.0)


def apply_gray_picker_balance(image: np.ndarray, point: tuple[int, int] | None) -> np.ndarray:
    if point is None:
        return image

    x, y = point
    h, w, _ = image.shape
    x = min(max(0, x), w - 1)
    y = min(max(0, y), h - 1)

    radius = 4
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    patch = image[y0:y1, x0:x1, :]
    sample = np.mean(patch.reshape(-1, 3), axis=0).astype(np.float32)

    target = float(np.mean(sample))
    gains = np.clip(target / np.maximum(sample, 1e-5), 0.75, 1.25)

    out = image * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def apply_temp_tint(image: np.ndarray, temp: float = 0.0, tint: float = 0.0) -> np.ndarray:
    """
    temp: warm/cool on red-blue axis
    tint: green-magenta on green channel
    """
    gains = np.array(
        [
            1.0 + temp * 0.12 + tint * 0.02,
            1.0 - tint * 0.10,
            1.0 - temp * 0.12 + tint * 0.02,
        ],
        dtype=np.float32,
    )
    out = image * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def adjust_saturation(image: np.ndarray, saturation: float = 0.0) -> np.ndarray:
    gray = np.mean(image, axis=2, keepdims=True)
    factor = 1.0 + saturation
    out = gray + (image - gray) * factor
    return np.clip(out, 0.0, 1.0)
