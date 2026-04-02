from __future__ import annotations

import numpy as np

from scanner.models.image_job import ImageJob
from scanner.core.image_io import read_image, resize_to_max_edge
from scanner.core.transforms import (
    apply_rotation_and_flips,
    crop_image,
    normalized_to_pixel_rect,
    pixel_to_normalized_rect,
    normalized_point_to_pixel,
)
from scanner.core.frame_detector import (
    detect_film_frame,
    estimate_content_mask,
    estimate_border_mask,
)
from scanner.core.negative import invert_color_negative, invert_bw_negative
from scanner.core.color import auto_balance, apply_gray_picker_balance, apply_temp_tint, adjust_saturation
from scanner.core.tone import (
    adjust_exposure,
    normalize_exposure_midtone,
    apply_levels,
    adjust_contrast,
    soft_highlight_rolloff,
    protect_extremes,
    render_border_soft,
    apply_filmic_contrast,
)
from scanner.core.sharpening import unsharp_mask
from scanner.core.histogram import compute_rgb_histograms


def resolve_crop_for_job(job: ImageJob, image: np.ndarray) -> tuple[int, int, int, int] | None:
    if job.normalized_crop_rect is not None:
        return normalized_to_pixel_rect(job.normalized_crop_rect, image.shape)

    if job.auto_crop_enabled:
        detected = detect_film_frame(image)
        if detected is not None:
            job.normalized_crop_rect = pixel_to_normalized_rect(detected, image.shape)
            return detected

    return None


def process_image(job: ImageJob, preview: bool = False) -> np.ndarray:
    image = read_image(job.source_path)

    if preview:
        image = resize_to_max_edge(image, 1600)

    image = apply_rotation_and_flips(image, job.rotation, job.flip_h, job.flip_v)

    crop_rect = resolve_crop_for_job(job, image)
    image = crop_image(image, crop_rect)

    # Build a strict safe-area mask for statistics
    content_mask = estimate_content_mask(
        image,
        crop_rect=None,
        include_border=False,
    )
    border_mask = estimate_border_mask(image, content_mask)

    # Film conversion should be driven by content-safe stats
    if job.film_type == "color_negative":
        image = invert_color_negative(
            image,
            border_hint=True,
            content_mask=content_mask,
        )
        image = normalize_exposure_midtone(image, content_mask)
        image = auto_balance(image, content_mask)

    elif job.film_type == "bw_negative":
        image = invert_bw_negative(image)
        image = normalize_exposure_midtone(image, content_mask)

    elif job.film_type == "slide_positive":
        image = normalize_exposure_midtone(image, content_mask)
        image = auto_balance(image, content_mask)

    # Optional gray pick after primary rendering
    gray_point = normalized_point_to_pixel(job.gray_pick_normalized, image.shape)
    image = apply_gray_picker_balance(image, gray_point)

    # User adjustments
    image = adjust_exposure(image, job.exposure)
    image = apply_temp_tint(image, job.temp, job.tint)
    image = apply_levels(image, job.black_point, job.white_point)
    image = adjust_contrast(image, job.contrast)
    image = protect_extremes(image)
    image = soft_highlight_rolloff(image, 0.15)
    image = apply_filmic_contrast(image)
    image = adjust_saturation(image, job.saturation)
    image = unsharp_mask(image, job.sharpness)

    # Border handled separately at the end, only if user wants to keep it
    if job.include_border:
        image = render_border_soft(image, border_mask)
    else:
        # If border is not requested, suppress contamination visually too
        if np.any(border_mask):
            neutral = np.mean(image, axis=2, keepdims=True).repeat(3, axis=2)
            image[border_mask] = neutral[border_mask] * 0.98

    return np.clip(image, 0.0, 1.0)


def process_image_and_histogram(
    job: ImageJob,
    preview: bool = False
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    image = process_image(job, preview=preview)
    hist = compute_rgb_histograms(image)
    return image, hist