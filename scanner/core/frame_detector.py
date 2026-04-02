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


def estimate_scene_mask(
    image: np.ndarray,
    crop_rect: CropRect | None = None,
    ignore_left_frac: float = 0.14,
    ignore_right_frac: float = 0.10,
    ignore_top_frac: float = 0.08,
    ignore_bottom_frac: float = 0.08,
) -> np.ndarray:
    """
    Build a strict scene-only mask.

    This intentionally rejects:
    - bright blown edge leaks
    - rebate / film border
    - low-detail contaminated edges
    - edge-connected near-white slabs
    """
    h, w = image.shape[:2]

    if crop_rect is None:
        crop_rect = detect_film_frame(image)

    if crop_rect is None:
        x0, y0, cw, ch = 0, 0, w, h
    else:
        x0, y0, cw, ch = crop_rect

    x1 = x0 + cw
    y1 = y0 + ch

    roi = image[y0:y1, x0:x1, :]
    rh, rw = roi.shape[:2]

    if rh < 16 or rw < 16:
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    gray = np.dot(roi[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    sat = (np.max(roi, axis=2) - np.min(roi, axis=2)).astype(np.float32)

    # 1) Start with a strong center-safe mask
    lx = int(rw * ignore_left_frac)
    rx = int(rw * ignore_right_frac)
    ty = int(rh * ignore_top_frac)
    by = int(rh * ignore_bottom_frac)

    center_safe = np.zeros((rh, rw), dtype=bool)
    center_safe[ty:rh - by, lx:rw - rx] = True

    # 2) Catastrophic leak rejection:
    # near-white / low-detail / edge-connected zones
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    detail = np.abs(lap)

    bright = blur > 0.94
    low_detail = detail < np.percentile(detail[center_safe], 35) if np.any(center_safe) else detail < np.percentile(detail, 35)
    catastrophic = bright & low_detail

    # Edge-connected catastrophic region
    edge_seed = np.zeros((rh, rw), dtype=np.uint8)
    edge_seed[0, :] = 1
    edge_seed[-1, :] = 1
    edge_seed[:, 0] = 1
    edge_seed[:, -1] = 1

    catastrophic_u8 = catastrophic.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(catastrophic_u8)

    edge_connected = np.zeros_like(catastrophic, dtype=bool)
    for label in range(1, num_labels):
        region = labels == label
        if np.any(region & (edge_seed > 0)):
            edge_connected |= region

    # 3) Reject contamination band around catastrophic region
    edge_connected_u8 = edge_connected.astype(np.uint8) * 255
    contamination_band = cv2.dilate(edge_connected_u8, np.ones((21, 21), np.uint8), iterations=1) > 0

    # 4) Reject likely border-like areas:
    # low saturation + similar to edge brightness
    band_y = max(2, rh // 20)
    band_x = max(2, rw // 20)
    border_samples = np.concatenate([
        blur[:band_y, :].reshape(-1),
        blur[-band_y:, :].reshape(-1),
        blur[:, :band_x].reshape(-1),
        blur[:, -band_x:].reshape(-1),
    ])

    border_med = float(np.median(border_samples)) if border_samples.size else float(np.median(blur))
    border_std = float(np.std(border_samples)) if border_samples.size else 0.02

    similar_to_border = np.abs(blur - border_med) < max(0.02, border_std * 0.7)
    low_sat = sat < (np.percentile(sat[center_safe], 55) if np.any(center_safe) else np.percentile(sat, 55))
    border_like = similar_to_border & low_sat

    # 5) Final local mask
    local_mask = center_safe & (~contamination_band) & (~border_like)

    # Morphology cleanup
    kernel = np.ones((5, 5), np.uint8)
    local_mask = cv2.morphologyEx(local_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel) > 0
    local_mask = cv2.morphologyEx(local_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0

    # 6) Fallback if too aggressive
    min_keep = max(1024, (rh * rw) // 10)
    if int(np.count_nonzero(local_mask)) < min_keep:
        fallback = np.zeros((rh, rw), dtype=bool)
        pad_y = max(4, rh // 10)
        pad_x = max(4, rw // 10)
        fallback[pad_y:rh - pad_y, pad_x:rw - pad_x] = True
        local_mask = fallback

    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = local_mask
    return mask


def estimate_border_mask(image: np.ndarray, scene_mask: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if scene_mask.shape != (h, w):
        raise ValueError("scene_mask shape mismatch")

    border_mask = ~scene_mask
    border_mask = cv2.morphologyEx(
        border_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    ) > 0

    return border_mask