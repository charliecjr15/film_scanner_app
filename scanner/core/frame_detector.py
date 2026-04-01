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
        margin_y = max(4, h // 20)
        margin_x = max(4, w // 20)
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
    gray = np.dot(roi[..., :3], [0.299, 0.587, 0.114])
    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)

    border_samples = np.concatenate([
        blur[: max(2, blur.shape[0] // 20), :].reshape(-1),
        blur[-max(2, blur.shape[0] // 20):, :].reshape(-1),
        blur[:, : max(2, blur.shape[1] // 20)].reshape(-1),
        blur[:, -max(2, blur.shape[1] // 20):].reshape(-1),
    ])

    if border_samples.size > 0:
        border_median = float(np.median(border_samples))
        border_distance = np.abs(blur - border_median)
        active = border_distance > max(0.03, np.std(border_samples) * 0.75)
    else:
        active = np.ones_like(blur, dtype=bool)

    active = cv2.morphologyEx(active.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    inner_mask = np.zeros_like(mask)
    inner_mask[y0:y1, x0:x1] = active

    min_keep = max(256, (y1 - y0) * (x1 - x0) // 6)
    if int(np.count_nonzero(inner_mask)) < min_keep:
        inner_y_pad = max(2, (y1 - y0) // 16)
        inner_x_pad = max(2, (x1 - x0) // 16)
        inner_mask[:] = False
        inner_mask[y0 + inner_y_pad:y1 - inner_y_pad, x0 + inner_x_pad:x1 - inner_x_pad] = True

    return inner_mask