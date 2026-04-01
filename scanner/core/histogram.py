from __future__ import annotations
import numpy as np

def compute_rgb_histograms(image: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    hist_r, _ = np.histogram(rgb8[:, :, 0], bins=bins, range=(0, 255))
    hist_g, _ = np.histogram(rgb8[:, :, 1], bins=bins, range=(0, 255))
    hist_b, _ = np.histogram(rgb8[:, :, 2], bins=bins, range=(0, 255))
    return hist_r.astype(np.float32), hist_g.astype(np.float32), hist_b.astype(np.float32)