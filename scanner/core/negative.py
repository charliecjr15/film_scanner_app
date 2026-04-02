from __future__ import annotations
import numpy as np

DEFAULT_FILM_BASE = np.array([0.82, 0.60, 0.42], dtype=np.float32)


def estimate_film_base_from_borders(
    image: np.ndarray,
    content_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate film base from border/rebate only.
    If content_mask exists, use its inverse; otherwise use fallback edges.
    """
    h, w, _ = image.shape

    if content_mask is not None and content_mask.shape[:2] == image.shape[:2] and np.any(content_mask):
        border_mask = ~content_mask
        border_pixels = image[border_mask]
    else:
        bh = max(6, h // 24)
        bw = max(6, w // 24)
        border_pixels = np.concatenate([
            image[:bh, :, :].reshape(-1, 3),
            image[-bh:, :, :].reshape(-1, 3),
            image[:, :bw, :].reshape(-1, 3),
            image[:, -bw:, :].reshape(-1, 3),
        ], axis=0)

    if border_pixels.size == 0:
        return DEFAULT_FILM_BASE.copy()

    # Bias toward denser / cleaner border pixels
    lo = np.percentile(border_pixels, 55, axis=0)
    hi = np.percentile(border_pixels, 97, axis=0)
    base = (lo * 0.30 + hi * 0.70).astype(np.float32)
    return np.clip(base, 0.05, 0.98)


def invert_color_negative(
    image: np.ndarray,
    border_hint: bool = True,
    content_mask: np.ndarray | None = None,
) -> np.ndarray:
    base = (
        estimate_film_base_from_borders(image, content_mask=content_mask)
        if border_hint else DEFAULT_FILM_BASE.copy()
    )

    normalized = image / np.maximum(base.reshape(1, 1, 3), 1e-5)
    normalized = np.clip(normalized, 0.0, 1.75)

    pos = 1.0 - np.clip(normalized, 0.0, 1.0)

    sample = pos[content_mask] if content_mask is not None and np.any(content_mask) else pos.reshape(-1, 3)
    if sample.shape[0] < 128:
        sample = pos.reshape(-1, 3)

    lo = np.percentile(sample, 1.0, axis=0)
    hi = np.percentile(sample, 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-5)
    pos = (pos - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    pos = np.clip(pos, 0.0, 1.0)

    # Gentle shadow de-coloring
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