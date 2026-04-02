from __future__ import annotations
import numpy as np

DEFAULT_FILM_BASE = np.array([0.82, 0.60, 0.42], dtype=np.float32)

NEGATIVE_PRESETS: dict[str, dict[str, np.ndarray | float]] = {
    "Balanced": {
        "base_bias": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "shadow_neutral": 0.18,
    },
    "Neutral Lab": {
        "base_bias": np.array([0.98, 1.00, 1.03], dtype=np.float32),
        "shadow_neutral": 0.12,
    },
    "Kodak Gold": {
        "base_bias": np.array([1.03, 1.00, 0.95], dtype=np.float32),
        "shadow_neutral": 0.16,
    },
    "Kodak Portra 400": {
        "base_bias": np.array([1.01, 1.00, 0.97], dtype=np.float32),
        "shadow_neutral": 0.14,
    },
    "Fuji 400H": {
        "base_bias": np.array([0.97, 1.00, 1.05], dtype=np.float32),
        "shadow_neutral": 0.13,
    },
    "CineStill 800T": {
        "base_bias": np.array([0.94, 1.00, 1.08], dtype=np.float32),
        "shadow_neutral": 0.22,
    },
}


def list_negative_presets() -> list[str]:
    return list(NEGATIVE_PRESETS.keys())


def get_negative_preset(name: str | None) -> dict[str, np.ndarray | float]:
    if not name:
        return NEGATIVE_PRESETS["Balanced"]
    return NEGATIVE_PRESETS.get(name, NEGATIVE_PRESETS["Balanced"])


def estimate_film_base_from_borders(
    image: np.ndarray,
    scene_mask: np.ndarray | None = None,
    preset_name: str | None = None,
) -> np.ndarray:
    """
    Estimate base ONLY from non-scene pixels, but reject catastrophic bright junk.
    """
    h, w, _ = image.shape

    if scene_mask is not None and scene_mask.shape[:2] == image.shape[:2] and np.any(scene_mask):
        border_mask = ~scene_mask
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
        base = DEFAULT_FILM_BASE.copy()
    else:
        # Reject catastrophic overbright junk from base estimation
        luma = np.mean(border_pixels, axis=1)
        usable = border_pixels[luma < 0.96]
        if usable.shape[0] < 128:
            usable = border_pixels

        lo = np.percentile(usable, 55, axis=0)
        hi = np.percentile(usable, 95, axis=0)
        base = (lo * 0.35 + hi * 0.65).astype(np.float32)

    preset = get_negative_preset(preset_name)
    base_bias = np.asarray(preset["base_bias"], dtype=np.float32)
    base = base * base_bias

    return np.clip(base, 0.05, 0.98)


def invert_color_negative(
    image: np.ndarray,
    border_hint: bool = True,
    scene_mask: np.ndarray | None = None,
    preset_name: str | None = None,
) -> np.ndarray:
    preset = get_negative_preset(preset_name)

    base = (
        estimate_film_base_from_borders(
            image,
            scene_mask=scene_mask,
            preset_name=preset_name,
        )
        if border_hint else DEFAULT_FILM_BASE.copy()
    )

    normalized = image / np.maximum(base.reshape(1, 1, 3), 1e-5)
    normalized = np.clip(normalized, 0.0, 1.60)

    pos = 1.0 - np.clip(normalized, 0.0, 1.0)

    sample = pos[scene_mask] if scene_mask is not None and np.any(scene_mask) else pos.reshape(-1, 3)
    if sample.shape[0] < 256:
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