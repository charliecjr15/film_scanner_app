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


def estimate_content_mask(
    image: np.ndarray,
    crop_rect: CropRect | None = None,
    include_border: bool = False,
) -> np.ndarray:
    h, w = image.shape[:2]

    if crop_rect is None:
        crop_rect = detect_film_frame(image)

    if crop_rect is None:
        margin_y = max(4, h // 18)
        margin_x = max(4, w // 18)
        mask = np.zeros((h, w), dtype=bool)
        mask[margin_y:h - margin_y, margin_x:w - margin_x] = True
    else:
        x, y, cw, ch = crop_rect
        mask = np.zeros((h, w), dtype=bool)
        mask[y:y + ch, x:x + cw] = True

    if include_border:
        return mask

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return mask

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    roi = image[y0:y1, x0:x1, :]
    gray = np.dot(roi[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    border_band_y = max(2, blur.shape[0] // 18)
    border_band_x = max(2, blur.shape[1] // 18)
    border_samples = np.concatenate([
        blur[:border_band_y, :].reshape(-1),
        blur[-border_band_y:, :].reshape(-1),
        blur[:, :border_band_x].reshape(-1),
        blur[:, -border_band_x:].reshape(-1),
    ])

    if border_samples.size > 0:
        border_med = float(np.median(border_samples))
        border_std = float(np.std(border_samples))
        active = np.abs(blur - border_med) > max(0.02, border_std * 0.6)
    else:
        active = np.ones_like(blur, dtype=bool)

    active = cv2.morphologyEx(
        (active.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    ) > 0

    inner_mask = np.zeros_like(mask)
    inner_mask[y0:y1, x0:x1] = active

    minimum_keep = max(256, ((y1 - y0) * (x1 - x0)) // 6)
    if int(np.count_nonzero(inner_mask)) < minimum_keep:
        pad_y = max(2, (y1 - y0) // 18)
        pad_x = max(2, (x1 - x0) // 18)
        inner_mask[:] = False
        inner_mask[y0 + pad_y:y1 - pad_y, x0 + pad_x:x1 - pad_x] = True

    return inner_mask