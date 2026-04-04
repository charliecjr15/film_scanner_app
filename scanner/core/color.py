from __future__ import annotations
import numpy as np


def auto_balance(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    sample = image[scene_mask] if scene_mask is not None and np.any(scene_mask) else image.reshape(-1, 3)
    if sample.shape[0] < 64:
        return image

    means = np.mean(sample, axis=0).astype(np.float32)
    target = float(np.mean(means))
    gains = target / np.maximum(means, 1e-5)
    gains = np.clip(gains, 0.92, 1.08)

    out = image * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def apply_filmic_color_balance(
    image: np.ndarray,
    scene_mask: np.ndarray | None = None,
) -> np.ndarray:
    sample = image[scene_mask] if scene_mask is not None and np.any(scene_mask) else image.reshape(-1, 3)
    if sample.shape[0] < 128:
        return image

    luma = np.mean(sample, axis=1)
    mids = sample[(luma > 0.22) & (luma < 0.75)]
    if mids.shape[0] < 64:
        mids = sample

    means = np.mean(mids, axis=0).astype(np.float32)
    target = np.array([
        means[1] * 1.01,
        means[1] * 1.00,
        means[1] * 0.98,
    ], dtype=np.float32)

    gains = target / np.maximum(means, 1e-5)
    gains = np.clip(gains, 0.94, 1.08)

    out = image * gains.reshape(1, 1, 3)
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
        1.0 + temp * 0.10,
        1.0 + tint * 0.07,
        1.0 - temp * 0.10,
    ], dtype=np.float32)

    out = image * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def adjust_saturation(image: np.ndarray, saturation: float = 0.0) -> np.ndarray:
    gray = np.mean(image, axis=2, keepdims=True)
    factor = 1.0 + saturation * 0.45
    out = gray + (image - gray) * factor
    return np.clip(out, 0.0, 1.0)