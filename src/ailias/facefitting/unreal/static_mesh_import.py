"""Import an appearance revision into UE without disguising it as a COLMAP job or a MetaHuman."""

import json
import uuid
from pathlib import Path

import numpy as np

from ailias.facefitting.artifacts import inspect_head_appearance
from ailias.generation.unreal import asset_path, readiness
from ailias.resources import finalize_resources
from ailias.storage import atomic_json


def import_appearance(store, revision, editor, project, tools):
    source = inspect_head_appearance(store, revision)
    if source["integrity_problems"] or source["status"] != "awaiting_appearance_review":
        raise ValueError("A completed, unchanged appearance revision is required")
    if "head_only_mesh" not in source["artifacts"]:
        raise ValueError("This revision predates the separate hair-free Unreal export")
    root = store.resolve(str(Path("facefitting") / "unreal" / uuid.uuid4().hex))
    root.mkdir(parents=True, exist_ok=False)
    probe = readiness(editor, project)
    report = {
        "revision": revision, "status": "blocked" if probe["blockers"] else "running",
        "readiness": probe, "artifacts": {}, "metahuman_created": False,
        "assembled": False, "rig_verified": False, "likeness_verified": False,
        "source": store.reference_artifact(store.resolve(str(
            Path("facefitting") / "appearance" / revision / "manifest.json"))),
        "limitations": [
            "Static head import only; not MetaHuman conform, rigging or appearance approval",
            "UE=(GNM Z,-X,Y)*100; upright orientation is checked but physical scale is uncalibrated",
            "Separate hair cap/strands are excluded from the custom-head candidate",
            "OBJ materials are a preview transfer, not production MetaHuman skin or measured optics",
        ],
    }
    manifest = root / "manifest.json"
    atomic_json(manifest, report)
    if probe["blockers"]:
        return report
    from ailias.generation import unreal_worker
    worker = None
    receipt = root / "import-receipt.json"
    request = root / "request.json"
    destination = asset_path(f"/Game/Ailias/PhotoHeads/{store.root.name.replace('-', '_')}/"
                              f"R{revision.replace('-', '_')}_{root.name[:8]}")
    try:
        worker = store.copy_artifact(Path(unreal_worker.__file__), str(root.relative_to(store.root) / "worker"))
        atomic_json(request, {"operation": "import", "receipt": str(receipt),
                              "mesh": str(store.artifact_path(source["artifacts"]["head_only_mesh"])),
                              "destination": destination, "report_geometry": True})
        tools.paths["unreal"] = Path(probe["editor"])
        report["runtime"] = tools.run("unreal", "photo-head-import", [
            probe["project"], "-run=pythonscript", f"-script={store.artifact_path(worker)}",
            "-unattended", "-nop4", "-nosplash", "-NullRHI", "-stdout", "-FullStdOutLogOutput",
        ], root, environment={"AILIAS_UNREAL_REQUEST": str(request)})
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if result.get("status") != "mesh_imported" or not result.get("assets"):
            raise ValueError("Unreal did not confirm a textured static head import")
        with np.load(store.artifact_path(source["artifacts"]["geometry"]), allow_pickle=False) as data:
            expected = data["vertices"][:, [2, 0, 1]]*[100, -100, 100]
        geometry = result.get("mesh_geometry", [])
        if len(geometry) != 1 or not np.allclose(
                [geometry[0]["bounds_min_xyz"], geometry[0]["bounds_max_xyz"]],
                [expected.min(0), expected.max(0)], atol=1e-3, rtol=0):
            raise ValueError("Imported head orientation/scale differs from the declared coordinate handoff")
        for index, package in enumerate(result["assets"]):
            relative = asset_path(package).removeprefix("/Game/").split(".")[0]
            asset = Path(probe["project"]).parent / "Content" / Path(*relative.split("/"))
            report["artifacts"][f"asset_{index}"] = store.reference_artifact(asset.with_suffix(".uasset"))
            for suffix in (".ubulk", ".uptnl", ".uexp"):
                if asset.with_suffix(suffix).is_file():
                    report["artifacts"][f"asset_{index}{suffix}"] = store.reference_artifact(
                        asset.with_suffix(suffix))
        if inspect_head_appearance(store, revision)["integrity_problems"]:
            raise ValueError("Appearance source changed during Unreal import")
        report.update(status="static_head_imported_for_review", destination=destination, imported=result,
                      upright_orientation_verified=True)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                      project_may_contain_partial_outputs=True)
    finally:
        if worker:
            report["artifacts"]["worker"] = worker
        if request.is_file() and request.stat().st_size:
            report["artifacts"]["request"] = store.reference_artifact(request)
        if receipt.is_file() and receipt.stat().st_size:
            report["artifacts"]["receipt"] = store.reference_artifact(receipt)
        finalize_resources(store, report, manifest)
        atomic_json(manifest, report)
    return report
