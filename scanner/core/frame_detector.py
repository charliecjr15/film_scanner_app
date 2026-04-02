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
    """
    Return a conservative mask for content-only statistics.
    This is intentionally aggressive: it rejects rebate, edge fog,
    and bright contaminated borders.
    """
    h, w = image.shape[:2]

    if crop_rect is None:
        crop_rect = detect_film_frame(image)

    # Base ROI
    if crop_rect is None:
        y_pad = max(6, h // 14)
        x_pad = max(6, w // 14)
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[y_pad:h - y_pad, x_pad:w - x_pad] = True
    else:
        x, y, cw, ch = crop_rect
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[y:y + ch, x:x + cw] = True

    if include_border:
        return roi_mask

    ys, xs = np.where(roi_mask)
    if ys.size == 0 or xs.size == 0:
        return roi_mask

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    roi = image[y0:y1, x0:x1, :]
    gray = np.dot(roi[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    # Strong inward shrink to make a safe stats zone
    inner_pad_y = max(4, roi.shape[0] // 12)
    inner_pad_x = max(4, roi.shape[1] // 12)
    safe = np.zeros_like(gray, dtype=bool)
    safe[inner_pad_y:roi.shape[0] - inner_pad_y, inner_pad_x:roi.shape[1] - inner_pad_x] = True

    # Detect edge contamination by comparing against border bands
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    band_y = max(2, roi.shape[0] // 18)
    band_x = max(2, roi.shape[1] // 18)

    border_samples = np.concatenate([
        blur[:band_y, :].reshape(-1),
        blur[-band_y:, :].reshape(-1),
        blur[:, :band_x].reshape(-1),
        blur[:, -band_x:].reshape(-1),
    ])

    border_med = float(np.median(border_samples)) if border_samples.size else float(np.median(blur))
    border_std = float(np.std(border_samples)) if border_samples.size else 0.02

    # Distance from border look
    border_distance = np.abs(blur - border_med)

    # Saturation anomaly catches purple / magenta / cyan edge leaks
    sat = np.max(roi, axis=2) - np.min(roi, axis=2)
    sat_blur = cv2.GaussianBlur(sat.astype(np.float32), (5, 5), 0)

    # Reject pixels that look too similar to border contamination
    active = (
        (border_distance > max(0.025, border_std * 0.75)) |
        (sat_blur < np.percentile(sat_blur[safe], 80) if np.any(safe) else 0.0)
    )

    # Keep only safe + content-like regions
    mask_local = safe & active

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask_local = cv2.morphologyEx(mask_local.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel) > 0
    mask_local = cv2.morphologyEx(mask_local.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0

    # Fallback if we got too aggressive
    min_keep = max(512, (roi.shape[0] * roi.shape[1]) // 8)
    if int(np.count_nonzero(mask_local)) < min_keep:
        mask_local[:] = False
        fallback_pad_y = max(4, roi.shape[0] // 10)
        fallback_pad_x = max(4, roi.shape[1] // 10)
        mask_local[fallback_pad_y:roi.shape[0] - fallback_pad_y, fallback_pad_x:roi.shape[1] - fallback_pad_x] = True

    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y0:y1, x0:x1] = mask_local
    return full_mask


def estimate_border_mask(
    image: np.ndarray,
    content_mask: np.ndarray,
) -> np.ndarray:
    """
    Border/rebate mask used for separate rendering when border inclusion is enabled.
    """
    h, w = image.shape[:2]
    if content_mask.shape != (h, w):
        raise ValueError("content_mask shape mismatch")

    border_mask = ~content_mask

    # Suppress tiny islands
    border_mask = cv2.morphologyEx(
        border_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    ) > 0

    return border_mask