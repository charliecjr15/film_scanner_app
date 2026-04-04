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
    estimate_scene_mask,
)
from scanner.core.negative import (
    invert_color_negative,
    invert_bw_negative,
    estimate_orange_mask_from_point,
)
from scanner.core.color import (
    auto_balance,
    apply_filmic_color_balance,
    apply_gray_picker_balance,
    apply_temp_tint,
    adjust_saturation,
)
from scanner.core.tone import (
    adjust_exposure,
    normalize_exposure_midtone,
    apply_levels,
    adjust_contrast,
    recover_highlights,
    soft_highlight_rolloff,
    apply_filmic_contrast,
)
from scanner.core.lut import apply_lut_profile
from scanner.core.sharpening import unsharp_mask
from scanner.core.histogram import compute_rgb_histograms


def resolve_crop_for_job(job: ImageJob, image: np.ndarray):
    if job.normalized_crop_rect is not None:
        return normalized_to_pixel_rect(job.normalized_crop_rect, image.shape)

    if job.auto_crop_enabled:
        detected = detect_film_frame(image)
        if detected is not None:
            job.normalized_crop_rect = pixel_to_normalized_rect(detected, image.shape)
            return detected

    return None


def _resolve_roll_mask_sample(job: ImageJob, image: np.ndarray):
    normalized = getattr(job, "orange_mask_pick_normalized", None)
    if normalized is None:
        return None

    point = normalized_point_to_pixel(normalized, image.shape)
    return estimate_orange_mask_from_point(image, point)


def process_image(job: ImageJob, preview: bool = False):
    image = read_image(job.source_path)

    if preview:
        image = resize_to_max_edge(image, 1600)

    image = apply_rotation_and_flips(image, job.rotation, job.flip_h, job.flip_v)

    crop_rect = resolve_crop_for_job(job, image)
    image = crop_image(image, crop_rect)

    # 🔥 MUCH STRONGER MASK (key NLP difference)
    scene_mask = estimate_scene_mask(
        image,
        crop_rect=None,
        keep_fraction=0.70,   # was 0.55
    )

    if job.film_type == "color_negative":
        orange_mask_rgb = _resolve_roll_mask_sample(job, image)

        image = invert_color_negative(
            image,
            scene_mask=scene_mask,
            orange_mask_rgb=orange_mask_rgb,
        )

        # 🔥 NLP-style order
        image = normalize_exposure_midtone(image, scene_mask)
        image = auto_balance(image, scene_mask)
        image = apply_filmic_color_balance(image, scene_mask)

        # 🔥 highlight BEFORE contrast
        image = recover_highlights(image, 0.45)
        image = soft_highlight_rolloff(image, 0.10)

        image = apply_filmic_contrast(image)

        # 🔥 LUT LAST (critical)
        lut_name = getattr(job, "preset_name", "Neutral Lab")
        image = apply_lut_profile(image, lut_name)

    elif job.film_type == "bw_negative":
        image = invert_bw_negative(image)
        image = normalize_exposure_midtone(image, scene_mask)

    elif job.film_type == "slide_positive":
        image = normalize_exposure_midtone(image, scene_mask)
        image = auto_balance(image, scene_mask)

    gray_point = normalized_point_to_pixel(job.gray_pick_normalized, image.shape)
    image = apply_gray_picker_balance(image, gray_point)

    image = apply_temp_tint(image, job.temp, job.tint)
    image = adjust_exposure(image, job.exposure)
    image = apply_levels(image, job.black_point, job.white_point)
    image = adjust_contrast(image, job.contrast)
    image = adjust_saturation(image, job.saturation)
    image = unsharp_mask(image, job.sharpness)

    return np.clip(image, 0.0, 1.0)


def process_image_and_histogram(job: ImageJob, preview: bool = False):
    image = process_image(job, preview=preview)
    hist = compute_rgb_histograms(image)
    return image, hist