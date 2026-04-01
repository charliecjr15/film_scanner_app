from __future__ import annotations
import cv2
import numpy as np

def unsharp_mask(image: np.ndarray, amount: float = 0.25) -> np.ndarray:
    if amount <= 0:
        return image

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    out = image + (image - blurred) * amount
    return np.clip(out, 0.0, 1.0)