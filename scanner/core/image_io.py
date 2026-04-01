from __future__ import annotations

from pathlib import Path
import cv2
import imageio.v3 as iio
import numpy as np

try:
    import rawpy
except Exception:
    rawpy = None

# Expanded RAW support list, including older Canon CRW
RAW_EXTS = {
    ".dng",
    ".nef",
    ".cr2",
    ".cr3",
    ".crw",
    ".arw",
    ".raf",
    ".rw2",
    ".orf",
    ".pef",
    ".srw",
    ".erf",
    ".kdc",
    ".mos",
    ".3fr",
    ".iiq",
    ".mrw",
    ".x3f",
}

def _read_standard_image(path: str) -> np.ndarray:
    file_path = Path(path)
    image = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unsupported or unreadable image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0

def _read_raw_image(path: str) -> np.ndarray:
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
                user_flip=0
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open RAW file:\n{path}\n\n"
            f"Details: {exc}\n\n"
            "If this is a Canon CR3 and it still fails, your installed rawpy/libraw build "
            "may not support that specific CR3 variant."
        ) from exc

    return rgb.astype(np.float32) / 65535.0

def read_image(path: str) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = file_path.suffix.lower()
    if suffix in RAW_EXTS:
        return _read_raw_image(path)

    return _read_standard_image(path)

def resize_to_max_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    current = max(h, w)
    if current <= max_edge:
        return image

    scale = max_edge / current
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def save_image_jpeg(path: str, image: np.ndarray, quality: int = 95) -> None:
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    iio.imwrite(path, rgb, extension=".jpg", quality=quality)

def save_image_tiff(path: str, image: np.ndarray) -> None:
    rgb = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    iio.imwrite(path, rgb, extension=".tiff")