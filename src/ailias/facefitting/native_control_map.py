"""Associate native sculpt indices with installed topology metadata, not another mesh."""

import json
import uuid
from pathlib import Path

import numpy as np

from ailias.storage import atomic_json


POSITION_TOLERANCE_CM = 1e-5
HEAD_MESH = "head_lod0_mesh"


def _vertex_id(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("Native metadata vertex IDs must address the exported topology")
    return value


def associate_controls(controls, vertices, configuration, anatomy):
    """Require coincident canonical vertices; never infer ordering or nearest anatomy."""
    points, surface = np.asarray(controls, dtype=float), np.asarray(vertices, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or surface.ndim != 2 or surface.shape[1] != 3 or not len(surface)
            or not np.isfinite(points).all() or not np.isfinite(surface).all()):
        raise ValueError("Native control and topology coordinates require finite nonempty Nx3 arrays")
    if not isinstance(configuration, dict) or not isinstance(anatomy, dict):
        raise ValueError("Native configuration and anatomy must be objects")
    definitions = {}
    for name, item in configuration.items():
        if not isinstance(item, dict):
            raise ValueError("Invalid native control configuration entry")
        if item.get("mesh") != HEAD_MESH or item.get("type") != "landmark":
            continue
        vertex = _vertex_id(item.get("vID"), len(surface))
        definitions.setdefault(vertex, []).append(name)
    if not definitions:
        raise ValueError("No native head control definitions")
    regions, inherited_meshes = {}, {}
    for name, item in anatomy.items():
        if not isinstance(item, dict):
            raise ValueError("Invalid native anatomical entry")
        if item.get("type") != "curve":
            continue
        for endpoint in ("start", "end"):
            if endpoint in item and item.get("mesh"):
                inherited_meshes.setdefault(item[endpoint], set()).add(item["mesh"])
        if item.get("mesh") != HEAD_MESH:
            continue
        for vertex in item.get("vIDs", []):
            regions.setdefault(_vertex_id(vertex, len(surface)), []).append(name)
    for name, item in anatomy.items():
        if item.get("type") != "landmark":
            continue
        mesh = item.get("mesh")
        if mesh != HEAD_MESH and not (
                mesh is None and inherited_meshes.get(name) == {HEAD_MESH}):
            continue
        regions.setdefault(_vertex_id(item.get("vID"), len(surface)), []).append(name)
    canonical = np.array(sorted(definitions), dtype=int)
    distances = np.linalg.norm(points[:, None, :]-surface[canonical][None, :, :], axis=2)
    records = []
    for index, distance in enumerate(distances):
        matches = canonical[distance <= POSITION_TOLERANCE_CM]
        if len(matches) != 1:
            records.append({
                "index": index, "status": "review",
                "reason": "no_coincident_definition" if not len(matches) else "ambiguous_coincident_definitions",
                "candidate_vertex_ids": matches.tolist(),
            })
            continue
        vertex = int(matches[0])
        records.append({
            "index": index, "status": "mapped", "vertex_id": vertex,
            "configuration_names": sorted(definitions[vertex]),
            "position_error_cm": float(np.linalg.norm(points[index]-surface[vertex])),
            "anatomical_memberships": sorted(set(regions.get(vertex, []))),
        })
    assigned = [item["vertex_id"] for item in records if item["status"] == "mapped"]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Multiple native controls resolve to the same canonical vertex")
    return {
        "status": "mapped" if len(assigned) == len(points) else "review",
        "controls": records, "reference_landmarks": points.tolist(),
        "summary": {
            "control_count": len(points), "mapped": len(assigned),
            "with_anatomical_membership": sum(bool(item.get("anatomical_memberships"))
                                            for item in records),
        },
        "position_tolerance_cm": POSITION_TOLERANCE_CM,
        "automatic_sculpt_proposal": False,
        "limitations": [
            "Same-native topology association only; no source-mesh wrapping or reconstructed geometry",
            "Configuration names are not anatomical labels; missing memberships remain unknown",
            "Curve membership is an exact vertex association, not a landmark correspondence to MediaPipe",
            "No control movements, identity fit or visual likeness approval",
        ],
    }


def map_native_controls(store, topology_path, inspection_path, configuration_path,
                        anatomy_path, source_package):
    """Retain a source-pinned map from an unchanged, read-only native inspection."""
    paths = (topology_path, inspection_path, configuration_path, anatomy_path, source_package)
    dependencies = [store.reference_artifact(path) for path in paths]
    topology, inspection, configuration, anatomy = [
        json.loads(path.read_text(encoding="utf-8-sig")) for path in paths[:4]]
    if (inspection.get("status") != "exported"
            or inspection.get("source_unchanged") is not True
            or inspection.get("source_saved") is not False
            or inspection.get("source_sha256") != dependencies[-1]["sha256"]):
        raise ValueError("Require an unchanged source package matching the native inspection")
    vertices = topology["vertices_cm"]
    if inspection.get("topology_vertices") != len(vertices):
        raise ValueError("Native inspection and exported topology disagree")
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Native control mapping evidence changed")
    report = associate_controls(inspection["landmarks_cm"], vertices, configuration, anatomy)
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Native control mapping evidence changed during association")
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("facefitting") / "native-control-maps" / revision))
    root.mkdir(parents=True, exist_ok=False)
    path = root / "manifest.json"
    report.update(schema_version=1, revision=revision, input_dependencies=dependencies,
                  native_mesh=topology["mesh"], manifest=str(path))
    atomic_json(path, report)
    return report
