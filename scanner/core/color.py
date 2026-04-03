from __future__ import annotations
import numpy as np


def auto_balance(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    sample = image[scene_mask] if scene_mask is not None and np.any(scene_mask) else image.reshape(-1, 3)

    if sample.shape[0] < 64:
        return image

    lo = np.percentile(sample, 2, axis=0)
    hi = np.percentile(sample, 98, axis=0)
    span = np.maximum(hi - lo, 1e-5)

    norm = (image - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    norm = np.clip(norm, 0.0, 1.0)

    stats_sample = norm[scene_mask] if scene_mask is not None and np.any(scene_mask) else norm.reshape(-1, 3)
    means = np.mean(stats_sample, axis=0)
    target = np.mean(means)
    gains = target / np.maximum(means, 1e-5)

    out = norm * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def apply_filmic_color_balance(
    image: np.ndarray,
    scene_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Midtone-focused filmic color rebalance.

    Purpose:
    - reduce green/cyan dominance after inversion
    - warm up building/skin-like midtones slightly
    - keep highlights and shadows more stable
    - move the rendering toward a more photographic feel
    """

    sample = image[scene_mask] if scene_mask is not None and np.any(scene_mask) else image.reshape(-1, 3)
    if sample.shape[0] < 128:
        return image

    luma = np.mean(sample, axis=1)
    mid_mask = (luma > 0.25) & (luma < 0.75)
    mids = sample[mid_mask] if np.any(mid_mask) else sample

    means = np.mean(mids, axis=0).astype(np.float32)

    # Encourage a warmer balance:
    # - slightly lift red
    # - hold green as reference
    # - slightly reduce blue
    target = np.array([
        means[1] * 1.03,   # red slightly warmer
        means[1] * 1.00,   # green reference
        means[1] * 0.94,   # blue reduced
    ], dtype=np.float32)

    gains = target / np.maximum(means, 1e-5)
    gains = np.clip(gains, 0.90, 1.12)

    luma_full = np.mean(image, axis=2, keepdims=True)

    # Strongest effect in midtones, weaker in shadows/highlights
    shadows_to_mids = np.clip((luma_full - 0.18) / 0.22, 0.0, 1.0)
    mids_to_highs = 1.0 - np.clip((luma_full - 0.72) / 0.20, 0.0, 1.0)
    mid_weight = shadows_to_mids * mids_to_highs

    out = image * (1.0 + (gains.reshape(1, 1, 3) - 1.0) * mid_weight)

    # Gentle cross-channel warmth for midtones only
    warm_bias = np.concatenate([
        mid_weight * 0.015,   # red lift
        np.zeros_like(mid_weight),
        -mid_weight * 0.020,  # blue reduction
    ], axis=2)
    out = out + warm_bias

    return np.clip(out, 0.0, 1.0)


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
    gains = target / np.maximum(sample, 1e-5)

    out = image * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def apply_temp_tint(image: np.ndarray, temp: float = 0.0, tint: float = 0.0) -> np.ndarray:
    gains = np.array([
        1.0 + temp * 0.12,
        1.0 + tint * 0.08,
        1.0 - temp * 0.12,
    ], dtype=np.float32)

    out = image * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def adjust_saturation(image: np.ndarray, saturation: float = 0.0) -> np.ndarray:
    gray = np.mean(image, axis=2, keepdims=True)
    factor = 1.0 + saturation
    out = gray + (image - gray) * factor
    return np.clip(out, 0.0, 1.0)