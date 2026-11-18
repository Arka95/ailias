"""Run an isolated Unreal operation and retain its receipt even when the process fails."""

import json
from ailias.generation.tools import ToolError
from ailias.storage import atomic_json


EXPECTED_STATUSES = {
    "create": "head_conformed_awaiting_review", "verify": "fresh_character_reloaded",
    "export_uv": "native_skin_uvs_exported", "import_texture": "native_photo_texture_imported",
    "preview": "native_editor_preview_rendered",
}


def run_native_job(tools, project, worker, root, operation, parameters, report):
    request = root / f"{operation}-request.json"
    receipt = root / f"{operation}-receipt.json"
    atomic_json(request, {"operation": operation, "receipt": str(receipt), **parameters})
    process_error = None
    try:
        # Native geometry export synchronizes Content Browser and requires Slate.
        report[operation + "_runtime"] = tools.run("unreal", "fresh-video-" + operation, [
            project, f"-ExecutePythonScript={worker}",
            "-unattended", "-nop4", "-nosplash", "-RenderOffscreen", "-d3d12",
            "-stdout", "-FullStdOutLogOutput",
            "-ExecCmds=mh.Character.FromCustomMeshImportShowIterations 0",
        ], root, environment={"AILIAS_FRESH_VIDEO_REQUEST": str(request)})
    except (ToolError, OSError, TimeoutError) as error:
        process_error = error
    try:
        result = json.loads(receipt.read_text(encoding="utf-8-sig"))
        if not isinstance(result, dict):
            raise ValueError("Native receipt must be a JSON object")
        report[operation] = result
    except (OSError, ValueError) as error:
        if process_error is not None:
            raise RuntimeError(
                f"Native {operation} failed without a readable receipt: {process_error}") from error
        raise RuntimeError(f"Native {operation} returned no valid receipt: {receipt}") from error
    if process_error is not None:
        raise RuntimeError(
            f"Native {operation} failed: {result.get('error', str(process_error))}") from process_error
    if result.get("status") == "failed":
        raise RuntimeError(f"Native {operation} rejected the operation: {result.get('error', 'unknown')}")
    if result.get("status") != EXPECTED_STATUSES[operation]:
        raise RuntimeError(f"Native {operation} did not complete: {result.get('status', 'missing status')}")
    return result
