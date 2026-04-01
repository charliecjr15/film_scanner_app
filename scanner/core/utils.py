from __future__ import annotations
import numpy as np

def clamp01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)