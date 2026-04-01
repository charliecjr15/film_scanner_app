from __future__ import annotations
import cv2
import numpy as np

def apply_rotation_and_flips(
    image: np.ndarray,
    rotation: int = 0,
    flip_h: bool = False,
    flip_v: bool = False
) -> np.ndarray:
    out = image.copy()
    rot = rotation % 360

    if rot == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rot == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if flip_h:
        out = cv2.flip(out, 1)
    if flip_v:
        out = cv2.flip(out, 0)

    return out

def crop_image(image: np.ndarray, crop_rect: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop_rect is None:
        return image

    x, y, w, h = crop_rect
    ih, iw = image.shape[:2]

    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = max(1, min(w, iw - x))
    h = max(1, min(h, ih - y))

    return image[y:y+h, x:x+w].copy()

def normalized_to_pixel_rect(
    norm_rect: tuple[float, float, float, float] | None,
    image_shape: tuple[int, int, int]
) -> tuple[int, int, int, int] | None:
    if norm_rect is None:
        return None

    h, w = image_shape[:2]
    nx, ny, nw, nh = norm_rect
    x = int(nx * w)
    y = int(ny * h)
    cw = int(nw * w)
    ch = int(nh * h)
    return (x, y, cw, ch)

def pixel_to_normalized_rect(
    rect: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int, int]
) -> tuple[float, float, float, float] | None:
    if rect is None:
        return None

    h, w = image_shape[:2]
    x, y, cw, ch = rect
    return (
        max(0.0, min(1.0, x / w)),
        max(0.0, min(1.0, y / h)),
        max(0.0, min(1.0, cw / w)),
        max(0.0, min(1.0, ch / h)),
    )

def normalized_point_to_pixel(
    pt: tuple[float, float] | None,
    image_shape: tuple[int, int, int]
) -> tuple[int, int] | None:
    if pt is None:
        return None
    h, w = image_shape[:2]
    return (int(pt[0] * w), int(pt[1] * h))