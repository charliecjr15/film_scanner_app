from __future__ import annotations
import numpy as np


def estimate_film_base_from_borders(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    bh = max(6, h // 24)
    bw = max(6, w // 24)

    border = np.concatenate([
        image[:bh, :, :].reshape(-1, 3),
        image[-bh:, :, :].reshape(-1, 3),
        image[:, :bw, :].reshape(-1, 3),
        image[:, -bw:, :].reshape(-1, 3),
    ], axis=0)

    if border.size == 0:
        return np.array([0.82, 0.60, 0.42], dtype=np.float32)

    lo = np.percentile(border, 55, axis=0)
    hi = np.percentile(border, 97, axis=0)
    base = (lo * 0.35 + hi * 0.65).astype(np.float32)
    return np.clip(base, 0.05, 0.98)


def invert_color_negative(image: np.ndarray, border_hint: bool = True) -> np.ndarray:
    """
    Convert a color negative to a positive image.
    Designed for a fast, dependable starting point:
    - estimate film base from borders
    - normalize by base
    - invert
    - robust channel stretch
    - gentle shadow neutralization
    """
    base = (
        estimate_film_base_from_borders(image)
        if border_hint
        else np.array([0.82, 0.60, 0.42], dtype=np.float32)
    )

    # Normalize against estimated film base to partially cancel orange mask influence
    normalized = image / np.maximum(base.reshape(1, 1, 3), 1e-5)
    normalized = np.clip(normalized, 0.0, 2.0)

    # Invert
    pos = 1.0 - np.clip(normalized, 0.0, 1.0)

    # Robust channel stretch for a cleaner baseline
    flat = pos.reshape(-1, 3)
    lo = np.percentile(flat, 1.0, axis=0)
    hi = np.percentile(flat, 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-5)
    pos = (pos - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    pos = np.clip(pos, 0.0, 1.0)

    # Gentle shadow de-coloring to reduce red/blue cast swings in darker areas
    luma = np.mean(pos, axis=2, keepdims=True)
    gray = np.repeat(luma, 3, axis=2)
    shadow_weight = np.clip((0.35 - luma) / 0.35, 0.0, 1.0)
    pos = pos * (1.0 - shadow_weight * 0.18) + gray * (shadow_weight * 0.18)

    return np.clip(pos, 0.0, 1.0)


def invert_bw_negative(image: np.ndarray) -> np.ndarray:
    gray = np.mean(image, axis=2, keepdims=True)
    inv = 1.0 - gray

    lo = np.percentile(inv, 1.0)
    hi = np.percentile(inv, 99.0)
    inv = (inv - lo) / max(hi - lo, 1e-5)

    return np.repeat(np.clip(inv, 0.0, 1.0), 3, axis=2)