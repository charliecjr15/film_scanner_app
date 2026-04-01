from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

CropRect = Tuple[int, int, int, int]

def detect_film_frame(image: np.ndarray) -> Optional[CropRect]:
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
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