from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

CropRect = Tuple[int, int, int, int]


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)


def detect_film_frame(image: np.ndarray) -> Optional[CropRect]:
    gray = _rgb_to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_area = h * w
    best = None
    best_score = -1.0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < img_area * 0.12:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        aspect = cw / max(1, ch)
        if not (0.5 <= aspect <= 2.5):
            continue

        fill = area / img_area
        rect_bonus = 1.0 if len(approx) == 4 else 0.75

        margin_penalty = 0.0
        if x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2:
            margin_penalty = 0.15

        score = fill * rect_bonus - margin_penalty
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    return best


def build_stats_mask(
    image: np.ndarray,
    keep_fraction: float = 0.55,
) -> np.ndarray:
    """
    Edge-aware stats mask:
    - rejects outer border
    - rejects brightest and darkest edge zones
    - keeps central scene region
    """
    h, w = image.shape[:2]
    keep_fraction = float(np.clip(keep_fraction, 0.35, 0.85))

    margin_y = int(h * (1.0 - keep_fraction) / 2.0)
    margin_x = int(w * (1.0 - keep_fraction) / 2.0)

    mask = np.zeros((h, w), dtype=bool)
    mask[margin_y:h - margin_y, margin_x:w - margin_x] = True

    gray = np.mean(image, axis=2)
    lo = np.percentile(gray[mask], 3.0) if np.any(mask) else np.percentile(gray, 3.0)
    hi = np.percentile(gray[mask], 97.0) if np.any(mask) else np.percentile(gray, 97.0)

    tonal_mask = (gray >= lo) & (gray <= hi)
    return mask & tonal_mask


def estimate_scene_mask(
    image: np.ndarray,
    crop_rect: CropRect | None = None,
    keep_fraction: float = 0.55,
) -> np.ndarray:
    if crop_rect is None:
        crop_rect = detect_film_frame(image)

    h, w = image.shape[:2]

    if crop_rect is None:
        return build_stats_mask(image, keep_fraction=keep_fraction)

    x, y, cw, ch = crop_rect
    roi = image[y:y + ch, x:x + cw]
    roi_mask = build_stats_mask(roi, keep_fraction=keep_fraction)

    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y:y + ch, x:x + cw] = roi_mask
    return full_mask