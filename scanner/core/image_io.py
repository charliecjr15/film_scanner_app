from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms

try:
    import rawpy
except Exception:
    rawpy = None

RAW_EXTS = {
    ".dng", ".nef", ".cr2", ".cr3", ".crw", ".arw", ".raf", ".rw2",
    ".orf", ".pef", ".srw", ".erf", ".kdc", ".mos", ".3fr", ".iiq",
    ".mrw", ".x3f",
}

WORKING_PROFILE_NAME = "sRGB IEC61966-2.1"


def _profile_bytes_from_name(name: str = WORKING_PROFILE_NAME) -> bytes:
    safe_name = WORKING_PROFILE_NAME if name != WORKING_PROFILE_NAME else name
    profile = ImageCms.createProfile("sRGB")
    wrapped = ImageCms.ImageCmsProfile(profile)
    return wrapped.tobytes()


SRGB_PROFILE_BYTES = _profile_bytes_from_name()


def _ensure_rgb_uint8(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode == "RGB":
        return pil_img
    if pil_img.mode in {"RGBA", "P", "LA", "L"}:
        return pil_img.convert("RGB")
    return pil_img.convert("RGB")


def _convert_pil_to_working_rgb(
    pil_img: Image.Image,
    embedded_icc: bytes | None,
) -> Image.Image:
    pil_img = _ensure_rgb_uint8(pil_img)

    if not embedded_icc:
        return pil_img

    try:
        src_profile = ImageCms.ImageCmsProfile(BytesIO(embedded_icc))
        dst_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(
            pil_img,
            src_profile,
            dst_profile,
            outputMode="RGB",
        )
        return converted
    except Exception:
        return pil_img


def _read_standard_image(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    file_path = Path(path)
    with Image.open(file_path) as pil_img:
        embedded_icc = pil_img.info.get("icc_profile")
        pil_rgb = _convert_pil_to_working_rgb(pil_img, embedded_icc)
        array = np.asarray(pil_rgb, dtype=np.float32) / 255.0

    meta = {
        "source_icc": embedded_icc,
        "working_icc": SRGB_PROFILE_BYTES,
        "working_profile_name": WORKING_PROFILE_NAME,
        "source_kind": "standard",
    }
    return array, meta


def _read_raw_image(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    if rawpy is None:
        raise RuntimeError(
            "RAW support requires rawpy. Install it with:\n"
            "pip install rawpy"
        )

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
        raise RuntimeError(
            f"Failed to open RAW file:\n{path}\n\n"
            f"Details: {exc}\n\n"
            "If this is a Canon CR3 and it still fails, your installed rawpy/libraw build "
            "may not support that specific CR3 variant."
        ) from exc

    meta = {
        "source_icc": None,
        "working_icc": SRGB_PROFILE_BYTES,
        "working_profile_name": WORKING_PROFILE_NAME,
        "source_kind": "raw",
    }
    return rgb.astype(np.float32) / 65535.0, meta


def read_image_with_profile(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = file_path.suffix.lower()
    if suffix in RAW_EXTS:
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

    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    resized = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_image_jpeg(
    path: str,
    image: np.ndarray,
    quality: int = 95,
    icc_profile: bytes | None = SRGB_PROFILE_BYTES,
) -> None:
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    pil.save(
        path,
        format="JPEG",
        quality=int(np.clip(quality, 1, 100)),
        subsampling=0,
        icc_profile=icc_profile,
        optimize=True,
    )


def save_image_tiff(
    path: str,
    image: np.ndarray,
    icc_profile: bytes | None = SRGB_PROFILE_BYTES,
) -> None:
    rgb = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    pil = Image.fromarray(rgb, mode="RGB")
    pil.save(path, format="TIFF", icc_profile=icc_profile, compression="tiff_adobe_deflate")