from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict

from PIL import ImageCms


@dataclass(frozen=True)
class OutputProfileSpec:
    name: str
    kind: str  # builtin | file
    path: str | None = None


def _srgb_profile_bytes() -> bytes:
    profile = ImageCms.createProfile("sRGB")
    return ImageCms.ImageCmsProfile(profile).tobytes()


SRGB_PROFILE_BYTES = _srgb_profile_bytes()

# AdobeRGB / Display P3 require real ICC files. This registry supports them
# when the files are placed into config/profiles/.
DEFAULT_OUTPUT_PROFILE_SPECS: Dict[str, OutputProfileSpec] = {
    "sRGB IEC61966-2.1": OutputProfileSpec("sRGB IEC61966-2.1", "builtin", None),
    "Adobe RGB (1998)": OutputProfileSpec("Adobe RGB (1998)", "file", "config/profiles/AdobeRGB1998.icc"),
    "Display P3": OutputProfileSpec("Display P3", "file", "config/profiles/DisplayP3.icc"),
}


def get_output_profile_specs(project_root: Path | None = None) -> Dict[str, OutputProfileSpec]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    specs: Dict[str, OutputProfileSpec] = {}
    for name, spec in DEFAULT_OUTPUT_PROFILE_SPECS.items():
        if spec.kind == "builtin":
            specs[name] = spec
        elif spec.path is not None:
            specs[name] = OutputProfileSpec(name, spec.kind, str(project_root / spec.path))
    return specs


def list_output_profile_names(project_root: Path | None = None) -> list[str]:
    return list(get_output_profile_specs(project_root).keys())


def load_output_profile_bytes(profile_name: str, custom_profile_path: str | None = None) -> bytes:
    if custom_profile_path:
        custom = Path(custom_profile_path)
        if custom.exists() and custom.is_file():
            return custom.read_bytes()

    specs = get_output_profile_specs()
    spec = specs.get(profile_name, specs["sRGB IEC61966-2.1"])

    if spec.kind == "builtin":
        return SRGB_PROFILE_BYTES

    if spec.path:
        path = Path(spec.path)
        if path.exists() and path.is_file():
            return path.read_bytes()

    return SRGB_PROFILE_BYTES


def build_transform_from_embedded_to_working(embedded_icc: bytes | None):
    if not embedded_icc:
        return None

    try:
        src = ImageCms.ImageCmsProfile(BytesIO(embedded_icc))
        dst = ImageCms.createProfile("sRGB")
        return src, dst
    except Exception:
        return None