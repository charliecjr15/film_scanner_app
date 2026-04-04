from __future__ import annotations

import numpy as np

_EPS = 1e-6

# These names are kept because the UI imports them.
NEGATIVE_PRESETS: dict[str, dict[str, np.ndarray | float]] = {
    "Neutral Lab": {
        "channel_bias": np.array([1.00, 1.00, 1.00], dtype=np.float32),
    },
    "Portra": {
        "channel_bias": np.array([1.02, 1.00, 0.98], dtype=np.float32),
    },
    "Gold": {
        "channel_bias": np.array([1.04, 1.00, 0.96], dtype=np.float32),
    },
    "Fuji": {
        "channel_bias": np.array([0.99, 1.00, 1.03], dtype=np.float32),
    },
    # Compatibility aliases
    "Balanced": {
        "channel_bias": np.array([1.00, 1.00, 1.00], dtype=np.float32),
    },
    "Kodak Portra 400": {
        "channel_bias": np.array([1.02, 1.00, 0.98], dtype=np.float32),
    },
    "Kodak Gold": {
        "channel_bias": np.array([1.04, 1.00, 0.96], dtype=np.float32),
    },
    "Fuji 400H": {
        "channel_bias": np.array([0.99, 1.00, 1.03], dtype=np.float32),
    },
}


def list_negative_presets() -> list[str]:
    return list(NEGATIVE_PRESETS.keys())


def get_negative_preset(name: str | None) -> dict[str, np.ndarray | float]:
    if not name:
        return NEGATIVE_PRESETS["Neutral Lab"]
    return NEGATIVE_PRESETS.get(name, NEGATIVE_PRESETS["Neutral Lab"])


def _sample_pixels(image: np.ndarray, scene_mask: np.ndarray | None = None) -> np.ndarray:
    if (
        scene_mask is not None
        and scene_mask.shape[:2] == image.shape[:2]
        and np.any(scene_mask)
    ):
        sample = image[scene_mask]
        if sample.shape[0] >= 128:
            return sample
    return image.reshape(-1, 3)


def estimate_orange_mask_from_point(
    image: np.ndarray,
    point: tuple[int, int] | None,
    radius: int = 10,
) -> np.ndarray | None:
    if point is None:
        return None

    h, w, _ = image.shape
    x, y = point
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    patch = image[y0:y1, x0:x1, :]
    if patch.size == 0:
        return None

    return np.mean(patch.reshape(-1, 3), axis=0).astype(np.float32)


def estimate_orange_mask_auto(
    image: np.ndarray,
    scene_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate film base / orange mask from bright negative pixels.
    Mask is for stats only.
    """
    sample = _sample_pixels(image, scene_mask=None)  # intentional fallback to whole frame
    luma = np.mean(sample, axis=1)

    hi = sample[luma >= np.percentile(luma, 98.5)]
    if hi.shape[0] < 32:
        hi = sample

    base = np.percentile(hi, 50.0, axis=0).astype(np.float32)
    return np.clip(base, 0.02, 0.98)


def normalize_negative_from_mask(
    image: np.ndarray,
    orange_mask_rgb: np.ndarray,
    scene_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalize RAW-derived linear negative before inversion.
    """
    img = np.clip(image.astype(np.float32), _EPS, 1.0)
    mask_rgb = np.clip(orange_mask_rgb.astype(np.float32), 0.01, 0.99)

    # Remove orange mask in transmittance space
    img = img / mask_rgb.reshape(1, 1, 3)
    img = np.clip(img, _EPS, 1.0)

    # Scene-only stats
    sample = _sample_pixels(img, scene_mask)

    lo = np.percentile(sample, 1.0, axis=0).astype(np.float32)
    hi = np.percentile(sample, 99.0, axis=0).astype(np.float32)
    span = np.maximum(hi - lo, 1e-5)

    norm = (img - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    return np.clip(norm, 0.0, 1.0)


def invert_color_negative(
    image: np.ndarray,
    scene_mask: np.ndarray | None = None,
    orange_mask_rgb: np.ndarray | None = None,
    border_hint: bool = False,   # kept for compatibility
    preset_name: str | None = None,
) -> np.ndarray:
    """
    Stable full-frame inversion.
    """
    if orange_mask_rgb is None:
        orange_mask_rgb = estimate_orange_mask_auto(image, scene_mask)

    norm = normalize_negative_from_mask(image, orange_mask_rgb, scene_mask)
    pos = 1.0 - norm

    preset = get_negative_preset(preset_name)
    bias = np.asarray(preset["channel_bias"], dtype=np.float32).reshape(1, 1, 3)
    pos = pos * bias

    return np.clip(pos, 0.0, 1.0)


def invert_bw_negative(image: np.ndarray) -> np.ndarray:
    gray = np.mean(np.clip(image.astype(np.float32), 0.0, 1.0), axis=2, keepdims=True)

    lo = float(np.percentile(gray, 1.0))
    hi = float(np.percentile(gray, 99.0))
    span = max(hi - lo, 1e-5)

    norm = (gray - lo) / span
    pos = 1.0 - np.clip(norm, 0.0, 1.0)
    return np.repeat(pos, 3, axis=2)