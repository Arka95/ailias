"""Decode original photos with orientation and colour handling; callers own returned images."""

import io
from pathlib import Path
import warnings

from PIL import Image, ImageCms, ImageOps


def read_oriented_image(path: Path, *, preserve_alpha: bool = False) -> tuple[Image.Image, dict]:
    """Decode oriented, colour-managed pixels without retaining the source file handle.
    Preserve transparency only when requested; existing callers continue receiving RGB.
    """
    if path.suffix.lower() in {".heic", ".heif", ".hif"}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as error:
            raise ValueError(
                "HEIC requires the imageprocessing optional dependencies"
            ) from error
        register_heif_opener(thumbnails=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original)
            image.load()
            alpha = (image.convert("RGBA").getchannel("A")
                     if preserve_alpha and ("A" in image.getbands() or "transparency" in image.info)
                     else None)
            metadata = {
                "decoded_size": list(original.size),
                "oriented_size": list(image.size),
                "exif_orientation": original.getexif().get(274, 1),
                "primary_frame_only": getattr(original, "n_frames", 1) > 1,
                "color_conversion": "RGB8; no exposure or geometry correction",
            }
            icc = original.info.get("icc_profile")
            if icc:
                try:
                    image = ImageCms.profileToProfile(
                        image,
                        ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                        ImageCms.createProfile("sRGB"),
                        outputMode="RGB",
                    )
                    metadata["color_conversion"] = "ICC to sRGB8"
                except (ImageCms.PyCMSError, OSError, ValueError):
                    metadata["color_warning"] = (
                        "invalid_icc_profile; RGB conversion only"
                    )
            image = image.convert("RGB")
            if alpha is not None:
                image.putalpha(alpha)
                metadata["alpha_preserved"] = True
            image.info.clear()
            return image, metadata
