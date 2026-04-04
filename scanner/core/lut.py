from __future__ import annotations

import numpy as np


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _apply_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    flat = image.reshape(-1, 3)
    out = flat @ matrix.T
    return out.reshape(h, w, 3)


def _soft_toe_shoulder(x: np.ndarray, toe: float, shoulder: float, gamma: float) -> np.ndarray:
    x = _clip01(x)
    x = np.power(x, max(gamma, 1e-6))

    # mild toe lift
    x = x * (1.0 - toe) + (x * x * (3.0 - 2.0 * x)) * toe

    # mild shoulder compression
    if shoulder > 0:
        t = np.clip((x - 0.72) / 0.28, 0.0, 1.0)
        shoulder_curve = t * t * (3.0 - 2.0 * t)
        compression = 1.0 - shoulder_curve * shoulder
        x = x * compression

    return _clip01(x)


LUT_PROFILES: dict[str, dict[str, np.ndarray | float]] = {
    "Neutral Lab": {
        "matrix": np.array([
            [1.00, 0.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 0.00, 1.00],
        ], dtype=np.float32),
        "toe": 0.10,
        "shoulder": 0.18,
        "gamma": 0.98,
        "sat": 0.92,
    },
    "Portra": {
        "matrix": np.array([
            [1.03, -0.02, -0.01],
            [0.00, 1.00, 0.00],
            [-0.01, 0.00, 1.01],
        ], dtype=np.float32),
        "toe": 0.14,
        "shoulder": 0.22,
        "gamma": 0.96,
        "sat": 0.90,
    },
    "Gold": {
        "matrix": np.array([
            [1.05, -0.02, -0.02],
            [0.00, 1.00, 0.00],
            [-0.02, -0.01, 1.04],
        ], dtype=np.float32),
        "toe": 0.16,
        "shoulder": 0.20,
        "gamma": 0.95,
        "sat": 0.95,
    },
    "Fuji": {
        "matrix": np.array([
            [0.99, 0.00, 0.01],
            [-0.01, 1.01, 0.00],
            [0.01, 0.00, 1.04],
        ], dtype=np.float32),
        "toe": 0.12,
        "shoulder": 0.20,
        "gamma": 0.97,
        "sat": 0.90,
    },
}


def list_lut_profiles() -> list[str]:
    return list(LUT_PROFILES.keys())


def get_lut_profile(name: str | None) -> dict[str, np.ndarray | float]:
    if not name:
        return LUT_PROFILES["Neutral Lab"]
    return LUT_PROFILES.get(name, LUT_PROFILES["Neutral Lab"])


def apply_lut_profile(image: np.ndarray, profile_name: str | None) -> np.ndarray:
    profile = get_lut_profile(profile_name)

    out = _apply_matrix(image, np.asarray(profile["matrix"], dtype=np.float32))
    out = _clip01(out)

    out = _soft_toe_shoulder(
        out,
        toe=float(profile["toe"]),
        shoulder=float(profile["shoulder"]),
        gamma=float(profile["gamma"]),
    )

    sat = float(profile["sat"])
    gray = np.mean(out, axis=2, keepdims=True)
    out = gray + (out - gray) * sat

    return _clip01(out)