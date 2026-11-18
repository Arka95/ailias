"""Reject shader-platform mistakes and material fallback in native execution logs."""

import re

PLATFORM = re.compile(r'\bshaderplatform="([^"]+)"', re.IGNORECASE)
FUNCTION_THUMBNAIL = re.compile(
    r"/Material_Functions/(MF_skin_(?:fuzzApply|bakedInputs))\.\1:Material_\d+:"
    r"\s*Failed to compile Material", re.IGNORECASE)


def validate_shader_output(text, *, require_platform=False):
    platforms = PLATFORM.findall(text)
    if require_platform and (not platforms or platforms[-1] != "PCD3D_SM6"):
        raise ValueError("Native MetaHuman work requires the active PCD3D_SM6 shader platform")
    failed_material = any(
        re.search(r"Failed to compile Material", line, re.IGNORECASE)
        and not FUNCTION_THUMBNAIL.search(line) for line in text.splitlines())
    if failed_material or re.search(r"maximum ps_5_0 sampler register", text, re.IGNORECASE):
        raise ValueError("Native material compilation failed; refusing fallback/checkerboard output")


def validate_unreal_log(path):
    platform = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            validate_shader_output(line)
            matches = PLATFORM.findall(line)
            if matches:
                platform = matches[-1]
    if platform != "PCD3D_SM6":
        raise ValueError("Native MetaHuman work did not confirm the active PCD3D_SM6 shader platform")
