from __future__ import annotations
import numpy as np

NEGATIVE_PRESETS: dict[str, dict[str, np.ndarray | float]] = {
    "Balanced": {
        "channel_bias": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "shadow_neutral": 0.14,
    },
    "Neutral Lab": {
        "channel_bias": np.array([0.99, 1.00, 1.02], dtype=np.float32),
        "shadow_neutral": 0.10,
    },
    "Kodak Gold": {
        "channel_bias": np.array([1.03, 1.00, 0.96], dtype=np.float32),
        "shadow_neutral": 0.15,
    },
    "Kodak Portra 400": {
        "channel_bias": np.array([1.01, 1.00, 0.98], dtype=np.float32),
        "shadow_neutral": 0.12,
    },
    "Fuji 400H": {
        "channel_bias": np.array([0.98, 1.00, 1.03], dtype=np.float32),
        "shadow_neutral": 0.12,
    },
    "CineStill 800T": {
        "channel_bias": np.array([0.95, 1.00, 1.08], dtype=np.float32),
        "shadow_neutral": 0.18,
    },
}


def list_negative_presets() -> list[str]:
    return list(NEGATIVE_PRESETS.keys())


def get_negative_preset(name: str | None) -> dict[str, np.ndarray | float]:
    if not name:
        return NEGATIVE_PRESETS["Balanced"]
    return NEGATIVE_PRESETS.get(name, NEGATIVE_PRESETS["Balanced"])


def _robust_scene_channel_anchors(
    image: np.ndarray,
    scene_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive inversion anchors from the scene center only.
    This replaces edge-derived film base estimation.
    """
    sample = image[scene_mask] if scene_mask is not None and np.any(scene_mask) else image.reshape(-1, 3)

    if sample.shape[0] < 256:
        sample = image.reshape(-1, 3)

    lo = np.percentile(sample, 1.0, axis=0).astype(np.float32)
    hi = np.percentile(sample, 99.0, axis=0).astype(np.float32)

    span = np.maximum(hi - lo, 1e-5)
    return lo, span


def invert_color_negative(
    image: np.ndarray,
    border_hint: bool = True,  # kept for UI/API compatibility
    scene_mask: np.ndarray | None = None,
    preset_name: str | None = None,
) -> np.ndarray:
    """
    Scene-anchored inversion:
    - derive per-channel anchors from center scene only
    - normalize from scene percentiles
    - invert
    - apply gentle preset bias
    - apply mild shadow neutralization
    """
    preset = get_negative_preset(preset_name)
    lo, span = _robust_scene_channel_anchors(image, scene_mask)

    norm = (image - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    norm = np.clip(norm, 0.0, 1.0)

    pos = 1.0 - norm

    channel_bias = np.asarray(preset["channel_bias"], dtype=np.float32).reshape(1, 1, 3)
    pos = np.clip(pos * channel_bias, 0.0, 1.0)

    # Re-normalize gently from scene only after preset bias
    sample = pos[scene_mask] if scene_mask is not None and np.any(scene_mask) else pos.reshape(-1, 3)
    if sample.shape[0] < 256:
        sample = pos.reshape(-1, 3)

    lo2 = np.percentile(sample, 1.0, axis=0)
    hi2 = np.percentile(sample, 99.0, axis=0)
    span2 = np.maximum(hi2 - lo2, 1e-5)
    pos = (pos - lo2.reshape(1, 1, 3)) / span2.reshape(1, 1, 3)
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

    lo = np.percentile(gray, 1.0)
    hi = np.percentile(gray, 99.0)
    norm = (gray - lo) / max(hi - lo, 1e-5)
    inv = 1.0 - np.clip(norm, 0.0, 1.0)

    return np.repeat(inv, 3, axis=2).clip(0.0, 1.0)