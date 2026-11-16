"""Validate ordered manual handoffs and produce three-view baseline comparisons.

Receipts record human decisions; they do not certify automated reconstruction.
"""

import io
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from ailias.intake import analyze_image
from ailias.settings import Policy
from ailias.storage import STAGES, Store, invalidate

HANDOFFS = {
    "reconstruction": {"mesh": {".obj", ".ply", ".fbx"},
                       "texture": {".png", ".jpg", ".jpeg"}, "cameras": {".json"}},
    "cleanup": {"mesh": {".obj", ".fbx"}, "texture": {".png", ".jpg", ".jpeg"}},
    "metahuman": {"evidence": {".png", ".jpg", ".jpeg"}},
}
BASELINE_VIEWS = ("front", "three_quarter", "profile")


def check_prerequisites(store: Store, state: dict, stage: str, policy: Policy) -> None:
    for previous in STAGES[:STAGES.index(stage)]:
        record = state["stages"][previous]
        if record["status"] != "complete":
            raise ValueError(f"Complete {previous} before {stage}")
        for artifact in record.get("artifacts", {}).values():
            if not store.verify(artifact):
                raise ValueError(f"{previous} artifact changed; attach it again")
    report = state["stages"]["intake"].get("report", {})
    if report.get("policy") != asdict(policy):
        raise ValueError("Validation policy changed; run validate again")
    for photo_id in report.get("accepted", []):
        if not store.verify(state["photos"][photo_id]["artifact"]):
            raise ValueError("Photo evidence changed; run intake and validate again")


def validate_cameras(path: Path, accepted: list[str]) -> None:
    cameras = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cameras, dict) or set(cameras) != set(accepted):
        raise ValueError("Cameras must map every accepted photo ID to its recovered camera")
    for camera in cameras.values():
        if not isinstance(camera, dict):
            raise ValueError("Each camera must be an object")
        for name, size in (("world_to_camera", 4), ("intrinsics", 3)):
            matrix = camera.get(name)
            if (not isinstance(matrix, list) or len(matrix) != size
                    or any(not isinstance(row, list) or len(row) != size for row in matrix)
                    or any(isinstance(v, bool) or not isinstance(v, (float, int))
                           or not math.isfinite(v) for row in matrix for v in row)):
                raise ValueError(f"Camera {name} must be a finite {size}x{size} matrix")
        error = camera.get("reprojection_error_px")
        if (isinstance(error, bool) or not isinstance(error, (int, float))
                or not math.isfinite(error) or error < 0):
            raise ValueError("Each camera needs non-negative reprojection_error_px")


def attach(store: Store, policy: Policy, stage: str, artifacts: dict[str, Path],
           note: str) -> dict:
    if stage not in HANDOFFS:
        raise ValueError("Unsupported manual handoff stage")
    if not note.strip():
        raise ValueError("A manual completion note is required")
    contract = HANDOFFS[stage]
    if set(artifacts) != set(contract):
        raise ValueError(f"{stage} requires exactly: {', '.join(contract)}")
    state = store.load()
    check_prerequisites(store, state, stage, policy)
    for role, path in artifacts.items():
        if path.suffix.lower() not in contract[role]:
            raise ValueError(f"Unsupported {role} file format")
        if role in {"texture", "evidence"}:
            analyze_image(path, policy)
    if stage == "reconstruction":
        validate_cameras(artifacts["cameras"],
                         state["stages"]["intake"]["report"]["accepted"])
    managed = {role: store.copy_artifact(path, stage) for role, path in artifacts.items()}
    if stage == "reconstruction":
        validate_cameras(store.resolve(managed["cameras"]["path"]),
                         state["stages"]["intake"]["report"]["accepted"])
    record = {"status": "complete", "mode": "manual_attestation",
              "note": note.strip(), "artifacts": managed}
    if state["stages"][stage] != record:
        invalidate(state, stage, "Manual handoff changed")
        state["stages"][stage] = record
        store.save(state)
    return record


def baseline(store: Store, policy: Policy, pairs: dict, note: str) -> dict:
    if not note.strip():
        raise ValueError("A human comparison note is required")
    if set(pairs) != set(BASELINE_VIEWS):
        raise ValueError("Baseline requires front, three_quarter and profile pairs")
    state = store.load()
    check_prerequisites(store, state, "baseline", policy)
    sheet = Image.new("RGB", (960, 3 * 400), "white")
    draw = ImageDraw.Draw(sheet)
    accepted = set(state["stages"]["intake"]["report"]["accepted"])
    artifacts = {}
    for row, view in enumerate(BASELINE_VIEWS):
        photo_id, render_path = pairs[view]
        matches = [key for key in accepted if key.startswith(photo_id)]
        if not photo_id or len(matches) != 1:
            raise ValueError("Baseline reference must uniquely identify an accepted photo")
        photo = state["photos"][matches[0]]
        review = photo["review"]
        target_yaw = {"front": 0, "three_quarter": 45, "profile": 90}[view]
        if (abs(abs(review["yaw"]) - target_yaw) > policy.angle_tolerance
                or abs(review["pitch"]) > policy.angle_tolerance):
            raise ValueError(f"Baseline {view} reference does not match the reviewed camera angle")
        render = store.copy_artifact(render_path, "renders")
        paths = (store.artifact_path(photo["artifact"]), store.resolve(render["path"]))
        for column, path in enumerate(paths):
            analyze_image(path, policy)
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((460, 360))
                sheet.paste(image, (column * 480 + 10, row * 400 + 30))
            if column:
                artifacts[f"{view}_render"] = render
            else:
                artifacts[f"{view}_reference"] = photo["artifact"]
        draw.text((10, row * 400 + 8), f"{view}: reference | render", fill="black")
    # Encode to memory so comparison sheets use the same content-addressed store.
    with io.BytesIO() as buffer:
        sheet.save(buffer, format="PNG")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparison.png"
            path.write_bytes(buffer.getvalue())
            artifacts["comparison"] = store.copy_artifact(path, "reports")
    record = {"status": "complete", "mode": "human_visual_baseline",
              "note": note.strip(), "artifacts": artifacts,
              "limitations": "Not a geometric score or animation/likeness certification"}
    if state["stages"]["baseline"] != record:
        invalidate(state, "baseline", "Visual baseline changed")
        state["stages"]["baseline"] = record
        store.save(state)
    return record


def status(store: Store) -> dict:
    state = store.load()
    problems = []
    for photo_id, photo in state["photos"].items():
        if not store.verify(photo["artifact"]):
            problems.append(f"photo:{photo_id}")
    for stage, record in state["stages"].items():
        for role, artifact in record.get("artifacts", {}).items():
            if not store.verify(artifact):
                problems.append(f"{stage}:{role}")
    return {"session": store.root.name, "photos": len(state["photos"]),
            "stages": state["stages"], "integrity_problems": problems,
            "history_entries": len(state["history"]),
            "baseline_recorded": all(record["status"] == "complete"
                                     for record in state["stages"].values()) and not problems,
            "production_ready": False}
