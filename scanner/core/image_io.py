from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms

from scanner.core.color_management import SRGB_PROFILE_BYTES, build_transform_from_embedded_to_working, load_output_profile_bytes

try:
    import rawpy
except Exception:
    rawpy = None

RAW_EXTS = {
    ".dng", ".nef", ".cr2", ".cr3", ".crw", ".arw", ".raf", ".rw2",
    ".orf", ".pef", ".srw", ".erf", ".kdc", ".mos", ".3fr", ".iiq",
    ".mrw", ".x3f",
}


def _ensure_rgb(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def _convert_standard_to_working_rgb(pil_img: Image.Image, embedded_icc: bytes | None) -> Image.Image:
    pil_img = _ensure_rgb(pil_img)
    transform = build_transform_from_embedded_to_working(embedded_icc)
    if transform is None:
        return pil_img

    try:
        src, dst = transform
        return ImageCms.profileToProfile(pil_img, src, dst, outputMode="RGB")
    except Exception:
        return pil_img


def _read_standard_image(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    file_path = Path(path)
    with Image.open(file_path) as pil_img:
        embedded_icc = pil_img.info.get("icc_profile")
        pil_rgb = _convert_standard_to_working_rgb(pil_img, embedded_icc)
        image = np.asarray(pil_rgb, dtype=np.float32) / 255.0

    return image, {
        "source_kind": "standard",
        "source_icc": embedded_icc,
        "working_icc": SRGB_PROFILE_BYTES,
        "working_profile_name": "sRGB IEC61966-2.1",
    }


def _read_raw_image(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    if rawpy is None:
        raise RuntimeError("RAW support requires rawpy. Install it with: pip install rawpy")

    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16,
                gamma=(1, 1),
                user_flip=0,
                output_color=rawpy.ColorSpace.sRGB,
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to open RAW file:\n{path}\n\nDetails: {exc}") from exc

    return rgb.astype(np.float32) / 65535.0, {
        "source_kind": "raw",
        "source_icc": None,
        "working_icc": SRGB_PROFILE_BYTES,
        "working_profile_name": "sRGB IEC61966-2.1",
    }


def read_image_with_profile(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if file_path.suffix.lower() in RAW_EXTS:
        return _read_raw_image(path)
    return _read_standard_image(path)


def read_image(path: str) -> np.ndarray:
    image, _meta = read_image_with_profile(path)
    return image


def resize_to_max_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    current = max(h, w)
    if current <= max_edge:
        return image

    scale = max_edge / current
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_image_jpeg(
    path: str,
    image: np.ndarray,
    quality: int = 95,
    output_profile_name: str = "sRGB IEC61966-2.1",
    custom_output_icc_path: str | None = None,
) -> None:
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    icc_bytes = load_output_profile_bytes(output_profile_name, custom_output_icc_path)
    pil.save(
        path,
        format="JPEG",
        quality=int(np.clip(quality, 1, 100)),
        subsampling=0,
        optimize=True,
        icc_profile=icc_bytes,
    )


def save_image_tiff(
    path: str,
    image: np.ndarray,
    output_profile_name: str = "sRGB IEC61966-2.1",
    custom_output_icc_path: str | None = None,
) -> None:
    rgb = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    pil = Image.fromarray(rgb, mode="RGB")
    icc_bytes = load_output_profile_bytes(output_profile_name, custom_output_icc_path)
    pil.save(
        path,
        format="TIFF",
        compression="tiff_adobe_deflate",
        icc_profile=icc_bytes,
    )