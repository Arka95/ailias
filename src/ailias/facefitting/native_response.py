"""Retain source-pinned native geometry response measurements for local fitting."""

import json
from pathlib import Path
import uuid

import numpy as np

from ailias.facefitting.native_features import character_package
from ailias.facefitting.native_features_worker import native_asset
from ailias.storage import atomic_json


def measure_response(store, character, editor, project, tools, *, step_cm=.05, indices=None):
    """Measure an isolated unrigged Character; never apply an identity-fit proposal."""
    character = native_asset(character)
    source = character_package(project, character)
    dependencies = [store.reference_artifact(source)]
    baseline = store.resolve(str(Path("facefitting") / "working-baseline.json"))
    if baseline.is_file():
        selected = json.loads(baseline.read_text(encoding="utf-8-sig"))
        dependencies.extend([store.reference_artifact(baseline), *selected["artifacts"].values()])
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Native response source or protected baseline changed")
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("facefitting") / "responses" / revision))
    root.mkdir(parents=True, exist_ok=False)
    worker = store.copy_artifact(Path(__file__).with_name("native_response_worker.py"),
                                 str(root.relative_to(store.root) / "worker"))
    manifest = root / "manifest.json"
    receipt = root / "receipt.json"
    request = root / "request.json"
    report = {
        "revision": revision, "status": "running", "source_character": character,
        "input_dependencies": dependencies, "artifacts": {"worker": worker},
        "likeness_verified": False, "native_fit_applied": False,
        "limitations": [
            "Forward finite-difference local response, not an exact global deformation model",
            "Native state coefficients are restored opaquely, never optimized by guessed indices",
            "Photo correspondences, camera/expression inference and held-out likeness remain separate",
        ],
    }
    atomic_json(request, {
        "character": character, "source_package": str(source), "step_cm": step_cm,
        "indices": indices, "output": str(root), "receipt": str(receipt),
        "destination": f"/Game/Ailias/GeometryResponses/R{revision}",
    })
    atomic_json(manifest, report)
    try:
        tools.paths["unreal"] = Path(editor)
        # Native geometry export synchronizes Content Browser and requires Slate.
        report["runtime"] = tools.run("unreal", "native-response", [
            project, f"-ExecutePythonScript={store.artifact_path(worker)}",
            "-unattended", "-nop4", "-nosplash", "-RenderOffscreen", "-d3d12",
            "-stdout", "-FullStdOutLogOutput",
        ], root, environment={"AILIAS_NATIVE_RESPONSE_REQUEST": str(request)})
        result = json.loads(receipt.read_text(encoding="utf-8-sig"))
        report["native_result"] = result
        if result.get("status") != "native_response_measured" or not result.get("source_unchanged"):
            raise ValueError("Native mesh response or source restoration was not verified")
        count = result["vertex_count"]
        expected = {"reference.f32": count*3, "responses.f32": len(result["responses"])*count*3}
        if count <= 0 or len(result["responses"]) != len(result["indices"])*3:
            raise ValueError("Native response dimensions are incomplete")
        for name, elements in expected.items():
            path = root / name
            if path.stat().st_size != elements*4:
                raise ValueError("Native response binary dimensions differ")
            with path.open("rb") as stream:
                while values := stream.read(1024*1024):
                    if not np.isfinite(np.frombuffer(values, dtype="<f4")).all():
                        raise ValueError("Native response binary contains nonfinite geometry")
        if not all(store.verify(item) for item in dependencies):
            raise ValueError("Native response source or baseline changed during measurement")
        report["status"] = "native_response_measured_awaiting_geometry_fit"
    except (OSError, ValueError, TypeError, KeyError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    finally:
        for name in ("request.json", "receipt.json", "reference.f32", "responses.f32", "triangles.json"):
            if (root / name).is_file():
                report["artifacts"][name] = store.reference_artifact(root / name)
        if not all(store.verify(item) for item in dependencies):
            report.update(status="failed", error={"type": "ValueError",
                                                  "message": "Protected native response inputs changed"})
        atomic_json(manifest, report)
    return report
