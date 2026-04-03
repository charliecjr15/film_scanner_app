from __future__ import annotations

import numpy as np

NEGATIVE_PRESETS: dict[str, dict[str, np.ndarray | float]] = {
    "Balanced": {
        "channel_bias": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "shadow_neutral": 0.12,
        "midtone_lift": 1.04,
    },
    "Neutral Lab": {
        "channel_bias": np.array([0.99, 1.00, 1.02], dtype=np.float32),
        "shadow_neutral": 0.10,
        "midtone_lift": 1.02,
    },
    "Kodak Gold": {
        "channel_bias": np.array([1.03, 1.00, 0.96], dtype=np.float32),
        "shadow_neutral": 0.14,
        "midtone_lift": 1.05,
    },
    "Kodak Portra 400": {
        "channel_bias": np.array([1.01, 1.00, 0.98], dtype=np.float32),
        "shadow_neutral": 0.11,
        "midtone_lift": 1.04,
    },
    "Fuji 400H": {
        "channel_bias": np.array([0.98, 1.00, 1.03], dtype=np.float32),
        "shadow_neutral": 0.11,
        "midtone_lift": 1.03,
    },
    "CineStill 800T": {
        "channel_bias": np.array([0.95, 1.00, 1.08], dtype=np.float32),
        "shadow_neutral": 0.16,
        "midtone_lift": 1.02,
    },
}

_EPS = 1e-6


def list_negative_presets() -> list[str]:
    return list(NEGATIVE_PRESETS.keys())


def get_negative_preset(name: str | None) -> dict[str, np.ndarray | float]:
    if not name:
        return NEGATIVE_PRESETS["Balanced"]
    return NEGATIVE_PRESETS.get(name, NEGATIVE_PRESETS["Balanced"])


def _safe_clip_transmittance(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32), _EPS, 1.0)


def _scene_sample(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    if scene_mask is not None and scene_mask.shape[:2] == image.shape[:2] and np.any(scene_mask):
        sample = image[scene_mask]
        if sample.shape[0] >= 256:
            return sample
    return image.reshape(-1, 3)


def _collect_border_pixels(image: np.ndarray, border_frac: float = 0.06) -> np.ndarray:
    h, w = image.shape[:2]
    bh = max(2, int(round(h * border_frac)))
    bw = max(2, int(round(w * border_frac)))

    strips = [
        image[:bh, :, :].reshape(-1, 3),
        image[h - bh :, :, :].reshape(-1, 3),
        image[:, :bw, :].reshape(-1, 3),
        image[:, w - bw :, :].reshape(-1, 3),
    ]
    return np.concatenate(strips, axis=0)


def _estimate_film_base(
    image: np.ndarray,
    border_hint: bool = True,
    scene_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate film-base transmittance in the SAME domain as the input image.

    border_hint=True:
        Prefer border strips because they usually contain unexposed film base.
    border_hint=False:
        Fall back to a high-percentile scene estimate that is more stable when
        no border is present or the user excludes it.
    """
    img = _safe_clip_transmittance(image)

    if border_hint:
        sample = _collect_border_pixels(img, border_frac=0.06)
        if sample.shape[0] >= 128:
            # favor the brighter border transmittance while avoiding blown light leaks
            lo = np.percentile(sample, 55.0, axis=0)
            hi = np.percentile(sample, 97.0, axis=0)
            base = lo * 0.35 + hi * 0.65
            return np.clip(base.astype(np.float32), 0.55, 0.995)

    sample = _scene_sample(img, scene_mask)
    # no border: use a conservative high-transmittance estimate from the whole frame
    base = np.percentile(sample, 99.2, axis=0).astype(np.float32)
    return np.clip(base, 0.60, 0.995)


def normalize_density(
    density: np.ndarray,
    scene_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalize density to a positive-image working range.

    Higher density should become brighter in the positive. We normalize per
    channel using robust scene percentiles, then apply a gentle luminance lift
    so scans do not come out muddy.
    """
    sample = _scene_sample(density, scene_mask)

    d_min = np.percentile(sample, 0.8, axis=0).astype(np.float32)
    d_max = np.percentile(sample, 99.4, axis=0).astype(np.float32)
    span = np.maximum(d_max - d_min, 1e-5)

    norm = (density - d_min.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    norm = np.clip(norm, 0.0, 1.0)

    scene_norm = _scene_sample(norm, scene_mask)
    luma = scene_norm @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    median_luma = float(np.percentile(luma, 50.0))
    gain = np.clip(0.58 / max(median_luma, 1e-5), 0.90, 1.30)

    return np.clip(norm * gain, 0.0, 1.0)


def invert_color_negative(
    image: np.ndarray,
    border_hint: bool = True,
    scene_mask: np.ndarray | None = None,
    preset_name: str | None = None,
) -> np.ndarray:
    """
    Density-first color negative inversion.

    Steps:
    1. Clip to stable transmittance range.
    2. Estimate film-base transmittance.
    3. Convert both image and base to density.
    4. Remove film base in density space.
    5. Normalize density to a positive-image range.
    6. Apply restrained preset bias and shadow neutralization.
    """
    preset = get_negative_preset(preset_name)
    img = _safe_clip_transmittance(image)

    film_base = _estimate_film_base(img, border_hint=border_hint, scene_mask=scene_mask)

    density = -np.log(img)
    base_density = -np.log(film_base.reshape(1, 1, 3))

    # Remove orange mask / film base in density space.
    scene_density = np.maximum(density - base_density, 0.0)

    # Map density directly to brightness in the positive.
    pos = normalize_density(scene_density, scene_mask=scene_mask)

    channel_bias = np.asarray(preset["channel_bias"], dtype=np.float32).reshape(1, 1, 3)
    pos = np.clip(pos * channel_bias, 0.0, 1.0)

    # Gentle preset-dependent midtone lift.
    midtone_lift = float(preset.get("midtone_lift", 1.0))
    luma = np.mean(pos, axis=2, keepdims=True)
    mid_weight = np.clip((luma - 0.15) / 0.55, 0.0, 1.0) * (1.0 - np.clip((luma - 0.78) / 0.18, 0.0, 1.0))
    pos = np.clip(pos * (1.0 + (midtone_lift - 1.0) * mid_weight), 0.0, 1.0)

    # Slightly de-color deep shadows where film scans often get murky casts.
    gray = np.repeat(np.mean(pos, axis=2, keepdims=True), 3, axis=2)
    shadow_weight = np.clip((0.28 - luma) / 0.28, 0.0, 1.0)
    shadow_neutral = float(preset["shadow_neutral"])
    pos = pos * (1.0 - shadow_weight * shadow_neutral) + gray * (shadow_weight * shadow_neutral)

    return np.clip(pos, 0.0, 1.0)


def invert_bw_negative(image: np.ndarray) -> np.ndarray:
    gray = np.mean(_safe_clip_transmittance(image), axis=2, keepdims=True)
    density = -np.log(gray)
    d_min = float(np.percentile(density, 0.8))
    d_max = float(np.percentile(density, 99.4))
    norm = (density - d_min) / max(d_max - d_min, 1e-5)
    pos = np.clip(norm, 0.0, 1.0)
    return np.repeat(pos, 3, axis=2).astype(np.float32)
