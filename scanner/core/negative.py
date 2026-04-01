from __future__ import annotations
import numpy as np

DEFAULT_FILM_BASE = np.array([0.82, 0.60, 0.42], dtype=np.float32)

# Important honesty:
# These are calibrated stock heuristics, not measured spectrophotometric lab datasets.
# They provide a framework for stock-aware de-masking and density curves inside your app.
NEGATIVE_PRESETS: dict[str, dict[str, np.ndarray | float]] = {
    "Balanced": {
        "mask_bias": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "scene_gamma": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "density_to_scene": np.array([1.05, 1.00, 0.95], dtype=np.float32),
        "toe_strength": 0.08,
        "shoulder_strength": 0.08,
        "shadow_neutral": 0.18,
    },
    "Neutral Lab": {
        "mask_bias": np.array([0.99, 1.00, 1.03], dtype=np.float32),
        "scene_gamma": np.array([1.00, 1.00, 1.00], dtype=np.float32),
        "density_to_scene": np.array([1.02, 1.00, 0.98], dtype=np.float32),
        "toe_strength": 0.06,
        "shoulder_strength": 0.07,
        "shadow_neutral": 0.12,
    },
    "Kodak Gold": {
        "mask_bias": np.array([1.03, 1.00, 0.95], dtype=np.float32),
        "scene_gamma": np.array([0.98, 1.00, 1.04], dtype=np.float32),
        "density_to_scene": np.array([1.09, 1.00, 0.90], dtype=np.float32),
        "toe_strength": 0.10,
        "shoulder_strength": 0.10,
        "shadow_neutral": 0.16,
    },
    "Kodak Portra 400": {
        "mask_bias": np.array([1.01, 1.00, 0.97], dtype=np.float32),
        "scene_gamma": np.array([0.99, 1.00, 1.02], dtype=np.float32),
        "density_to_scene": np.array([1.05, 1.00, 0.94], dtype=np.float32),
        "toe_strength": 0.08,
        "shoulder_strength": 0.09,
        "shadow_neutral": 0.14,
    },
    "Fuji 400H": {
        "mask_bias": np.array([0.97, 1.00, 1.05], dtype=np.float32),
        "scene_gamma": np.array([1.02, 1.00, 0.98], dtype=np.float32),
        "density_to_scene": np.array([0.96, 1.00, 1.06], dtype=np.float32),
        "toe_strength": 0.07,
        "shoulder_strength": 0.07,
        "shadow_neutral": 0.13,
    },
    "CineStill 800T": {
        "mask_bias": np.array([0.94, 1.00, 1.08], dtype=np.float32),
        "scene_gamma": np.array([1.04, 1.00, 0.96], dtype=np.float32),
        "density_to_scene": np.array([0.93, 1.00, 1.10], dtype=np.float32),
        "toe_strength": 0.10,
        "shoulder_strength": 0.08,
        "shadow_neutral": 0.22,
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
    mask_bias = np.asarray(preset["mask_bias"], dtype=np.float32)
    return np.clip(base * mask_bias, 0.05, 0.98)


def _pseudo_spectral_deorange(
    image: np.ndarray,
    film_base: np.ndarray,
    preset_name: str | None = None,
) -> np.ndarray:
    preset = get_negative_preset(preset_name)
    density_bias = np.asarray(preset["density_to_scene"], dtype=np.float32).reshape(1, 1, 3)

    transmittance = image / np.maximum(film_base.reshape(1, 1, 3), 1e-5)
    transmittance = np.clip(transmittance, 1e-5, 1.0)

    # Density domain transform — pseudo-spectral approximation.
    density = -np.log10(transmittance)
    density = density * density_bias
    density = np.clip(density, 0.0, 3.5)

    scene = 10.0 ** (-density)
    return np.clip(scene, 0.0, 1.0)


def _apply_stock_sensitometric_curve(scene: np.ndarray, preset_name: str | None = None) -> np.ndarray:
    preset = get_negative_preset(preset_name)
    gamma = np.asarray(preset["scene_gamma"], dtype=np.float32).reshape(1, 1, 3)
    toe_strength = float(preset["toe_strength"])
    shoulder_strength = float(preset["shoulder_strength"])

    out = np.power(np.clip(scene, 0.0, 1.0), gamma)

    toe = np.clip((0.22 - out) / 0.22, 0.0, 1.0)
    shoulder = np.clip((out - 0.75) / 0.25, 0.0, 1.0)

    toe_lift = out + toe * toe_strength * (0.22 - out)
    shoulder_soft = 0.75 + (out - 0.75) * (1.0 - shoulder_strength * shoulder)
    out = np.where(out < 0.22, toe_lift, out)
    out = np.where(out > 0.75, shoulder_soft, out)

    return np.clip(out, 0.0, 1.0)


def invert_color_negative(
    image: np.ndarray,
    border_hint: bool = True,
    content_mask: np.ndarray | None = None,
    preset_name: str | None = None,
) -> np.ndarray:
    base = estimate_film_base_from_borders(image, content_mask=content_mask, preset_name=preset_name) if border_hint else DEFAULT_FILM_BASE.copy()

    # Pseudo-spectral orange-mask removal
    scene = _pseudo_spectral_deorange(image, base, preset_name=preset_name)
    pos = 1.0 - scene

    flat = pos[content_mask] if content_mask is not None and np.any(content_mask) else pos.reshape(-1, 3)
    if flat.shape[0] < 128:
        flat = pos.reshape(-1, 3)

    lo = np.percentile(flat, 1.0, axis=0)
    hi = np.percentile(flat, 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-5)
    pos = (pos - lo.reshape(1, 1, 3)) / span.reshape(1, 1, 3)
    pos = np.clip(pos, 0.0, 1.0)

    pos = _apply_stock_sensitometric_curve(pos, preset_name=preset_name)

    luma = np.mean(pos, axis=2, keepdims=True)
    gray = np.repeat(luma, 3, axis=2)
    shadow_weight = np.clip((0.35 - luma) / 0.35, 0.0, 1.0)
    shadow_neutral = float(get_negative_preset(preset_name)["shadow_neutral"])
    pos = pos * (1.0 - shadow_weight * shadow_neutral) + gray * (shadow_weight * shadow_neutral)

    return np.clip(pos, 0.0, 1.0)


def invert_bw_negative(image: np.ndarray) -> np.ndarray:
    gray = np.mean(image, axis=2, keepdims=True)
    inv = 1.0 - gray

    lo = np.percentile(inv, 1.0)
    hi = np.percentile(inv, 99.0)
    inv = (inv - lo) / max(hi - lo, 1e-5)

    return np.repeat(np.clip(inv, 0.0, 1.0), 3, axis=2)