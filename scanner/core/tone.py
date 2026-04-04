from __future__ import annotations
import numpy as np


def adjust_exposure(image: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    factor = 2.0 ** exposure
    return np.clip(image * factor, 0.0, 1.0)


def normalize_exposure_midtone(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    valid = gray[scene_mask] if scene_mask is not None and np.any(scene_mask) else gray.reshape(-1)
    if valid.size < 100:
        return image

    # 🔥 NLP-style midtone anchor (brighter)
    mid = np.percentile(valid, 50)
    target_mid = 0.55   # was 0.46 → HUGE difference

    gain = target_mid / (mid + 1e-6)
    gain = np.clip(gain, 0.9, 1.6)

    return np.clip(image * gain, 0.0, 1.0)


def apply_levels(image: np.ndarray, black_point: float = 0.0, white_point: float = 1.0) -> np.ndarray:
    white_point = max(white_point, black_point + 1e-5)
    out = (image - black_point) / (white_point - black_point)
    return np.clip(out, 0.0, 1.0)


def adjust_contrast(image: np.ndarray, contrast: float = 0.0) -> np.ndarray:
    factor = 1.0 + contrast * 0.45
    midpoint = 0.5
    out = (image - midpoint) * factor + midpoint
    return np.clip(out, 0.0, 1.0)


def recover_highlights(image: np.ndarray, strength: float = 0.35) -> np.ndarray:
    x = np.clip(image, 0.0, 1.0)
    luma = np.dot(x[..., :3], [0.2126, 0.7152, 0.0722])[..., None]

    threshold = 0.70
    t = np.clip((luma - threshold) / (1.0 - threshold), 0.0, 1.0)

    compression = 1.0 - (t * t) * strength
    out = x * compression

    gray = np.repeat(luma, 3, axis=2)
    out = out * (1.0 - t * 0.18) + gray * (t * 0.18)

    return np.clip(out, 0.0, 1.0)


def soft_highlight_rolloff(image: np.ndarray, strength: float = 0.06) -> np.ndarray:
    if strength <= 0:
        return image

    x = np.clip(image, 0.0, 1.0)
    luma = np.dot(x[..., :3], [0.2126, 0.7152, 0.0722])[..., None]

    shoulder_start = 0.78
    t = np.clip((luma - shoulder_start) / (1.0 - shoulder_start), 0.0, 1.0)
    t_curve = t * t * (3.0 - 2.0 * t)

    compression = 1.0 - t_curve * strength
    out = x * compression

    return np.clip(out, 0.0, 1.0)


def apply_filmic_contrast(image: np.ndarray) -> np.ndarray:
    image = np.power(np.clip(image, 0.0, 1.0), 0.98)
    image = image * image * (3.0 - 2.0 * image)
    return np.clip(image, 0.0, 1.0)