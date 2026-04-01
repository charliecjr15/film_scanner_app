from __future__ import annotations
import numpy as np

DEFAULT_FILM_BASE = np.array([0.82, 0.60, 0.42], dtype=np.float32)

NEGATIVE_PRESETS: dict[str, dict[str, np.ndarray | float]] = {
    "Balanced": {
        "base_bias": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "channel_gamma": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "shadow_neutral": 0.18,
        "contrast_hint": 1.00,
    },
    "Neutral Lab": {
        "base_bias": np.array([0.98, 1.00, 1.03], dtype=np.float32),
        "channel_gamma": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "shadow_neutral": 0.12,
        "contrast_hint": 0.96,
    },
    "Kodak Gold": {
        "base_bias": np.array([1.03, 1.00, 0.94], dtype=np.float32),
        "channel_gamma": np.array([0.98, 1.00, 1.04], dtype=np.float32),
        "shadow_neutral": 0.16,
        "contrast_hint": 1.05,
    },
    "Kodak Portra 400": {
        "base_bias": np.array([1.01, 1.00, 0.97], dtype=np.float32),
        "channel_gamma": np.array([0.99, 1.00, 1.02], dtype=np.float32),
        "shadow_neutral": 0.14,
        "contrast_hint": 0.98,
    },
    "Fuji 400H": {
        "base_bias": np.array([0.97, 1.00, 1.04], dtype=np.float32),
        "channel_gamma": np.array([1.02, 1.00, 0.98], dtype=np.float32),
        "shadow_neutral": 0.14,
        "contrast_hint": 0.95,
    },
    "CineStill 800T": {
        "base_bias": np.array([0.94, 1.00, 1.08], dtype=np.float32),
        "channel_gamma": np.array([1.04, 1.00, 0.96], dtype=np.float32),
        "shadow_neutral": 0.22,
        "contrast_hint": 1.02,
    },
}


def list_negative_presets() -> list[str]:
    return list(NEGATIVE_PRESETS.keys())


def get_negative_preset(name: str | None) -> dict[str, np.ndarray | float]:
    if not name:
        return NEGATIVE_PRESETS["Balanced"]
    return NEGATIVE_PRESETS.get(name, NEGATIVE_PRESETS["Balanced"])


def _fallback_border_pixels(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    bh = max(6, h // 24)
    bw = max(6, w // 24)
    return np.concatenate([
        image[:bh, :, :].reshape(-1, 3),
        image[-bh:, :, :].reshape(-1, 3),
        image[:, :bw, :].reshape(-1, 3),
        image[:, -bw:, :].reshape(-1, 3),
    ], axis=0)


def estimate_film_base_from_borders(
    image: np.ndarray,
    content_mask: np.ndarray | None = None,
    preset_name: str | None = None,
) -> np.ndarray:
    h, w, _ = image.shape
    border_ring = np.ones((h, w), dtype=bool)

    if content_mask is not None and content_mask.shape[:2] == image.shape[:2] and np.any(content_mask):
        border_ring = ~content_mask

    border_pixels = image[border_ring]
    if border_pixels.shape[0] < 256:
        border_pixels = _fallback_border_pixels(image)

    if border_pixels.size == 0:
        base = DEFAULT_FILM_BASE.copy()
    else:
        lo = np.percentile(border_pixels, 55, axis=0)
        hi = np.percentile(border_pixels, 97, axis=0)
        base = (lo * 0.35 + hi * 0.65).astype(np.float32)

    preset = get_negative_preset(preset_name)
    base_bias = np.asarray(preset["base_bias"], dtype=np.float32)
    base = base * base_bias
    return np.clip(base, 0.05, 0.98)


def invert_color_negative(
    image: np.ndarray,
    border_hint: bool = True,
    content_mask: np.ndarray | None = None,
    preset_name: str | None = None,
) -> np.ndarray:
    preset = get_negative_preset(preset_name)

    base = (
        estimate_film_base_from_borders(image, content_mask=content_mask, preset_name=preset_name)
        if border_hint else DEFAULT_FILM_BASE.copy()
    )

    normalized = image / np.maximum(base.reshape(1, 1, 3), 1e-5)
    normalized = np.clip(normalized, 0.0, 1.75)

    pos = 1.0 - np.clip(normalized, 0.0, 1.0)

    flat = pos[content_mask] if content_mask is not None and np.any(content_mask) else pos.reshape(-1, 3)
    if flat.shape[0] < 128:
        flat = pos.reshape(-1, 3)

    lo = np.percentile(flat, 1.0, axis=0)
    hi = np.percentile(flat, 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-5)
    pos = (pos - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    pos = np.clip(pos, 0.0, 1.0)

    channel_gamma = np.asarray(preset["channel_gamma"], dtype=np.float32).reshape(1, 1, 3)
    pos = np.power(np.clip(pos, 0.0, 1.0), channel_gamma)

    luma = np.mean(pos, axis=2, keepdims=True)
    gray = np.repeat(luma, 3, axis=2)
    shadow_weight = np.clip((0.35 - luma) / 0.35, 0.0, 1.0)
    shadow_neutral = float(preset["shadow_neutral"])
    pos = pos * (1.0 - shadow_weight * shadow_neutral) + gray * (shadow_weight * shadow_neutral)

    return np.clip(pos, 0.0, 1.0)


def invert_bw_negative(image: np.ndarray) -> np.ndarray:
    gray = np.mean(image, axis=2, keepdims=True)
    inv = 1.0 - gray

    lo = np.percentile(inv, 1.0)
    hi = np.percentile(inv, 99.0)
    inv = (inv - lo) / max(hi - lo, 1e-5)

    return np.repeat(np.clip(inv, 0.0, 1.0), 3, axis=2)