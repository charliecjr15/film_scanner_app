from __future__ import annotations
import numpy as np


def auto_balance(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    flat = image[mask] if mask is not None and np.any(mask) else image.reshape(-1, 3)

    if flat.shape[0] < 64:
        return image

    lo = np.percentile(flat, 2, axis=0)
    hi = np.percentile(flat, 98, axis=0)
    span = np.maximum(hi - lo, 1e-5)

    norm = (image - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    norm = np.clip(norm, 0.0, 1.0)

    sample = norm[mask] if mask is not None and np.any(mask) else norm.reshape(-1, 3)
    means = np.mean(sample, axis=0)
    target = np.mean(means)
    gains = target / np.maximum(means, 1e-5)

    out = norm * gains.reshape(1, 1, 3)
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
        1.0 - temp * 0.12
    ], dtype=np.float32)

    out = image * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def adjust_saturation(image: np.ndarray, saturation: float = 0.0) -> np.ndarray:
    gray = np.mean(image, axis=2, keepdims=True)
    factor = 1.0 + saturation
    out = gray + (image - gray) * factor
    return np.clip(out, 0.0, 1.0)