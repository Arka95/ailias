"""Compare referenced geometry or detected faces and retain auditable input snapshots.

Recognition similarity remains separate from geometric error measurements.
"""

import json
from pathlib import Path

from ailias.geometry.metrics import compare_geometry
from ailias.imageprocessing.features import similarity
from ailias.imageprocessing.pipeline import load_run, new_id
from ailias.resources import cleanup_owned_files
from ailias.storage import Store, atomic_json


def compare_detected_faces(store: Store, run: str, first: str, second: str) -> dict:
    report = load_run(store, run)
    selected = []
    for prefix in (first, second):
        matches = [face for key, face in report["faces"].items() if prefix and key.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError("Face ID prefix must uniquely identify a detected face")
        selected.append(matches[0])
    names = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")
    geometries = [
        {"dimension": 2, "units": "pixels",
         "landmarks": dict(zip(names, face["landmarks_5"]))}
        for face in selected
    ]
    result = {
        "run": run, "first": selected[0]["id"], "second": selected[1]["id"],
        "identity_cosine": max(-1.0, min(1.0, similarity(
            selected[0]["embedding"], selected[1]["embedding"]))),
        "geometry_2d": compare_geometry(*geometries, fit="similarity"),
        "warning": "Five projected points only; compare matched pose/expression/camera. Not a 3D likeness index.",
    }
    path = store.resolve(str(Path("imageprocessing") / "comparisons" / f"{new_id()}.json"))
    result["report"] = str(path)
    atomic_json(path, result)
    return result


def compare_files(store: Store, first: Path, second: Path, fit: str) -> dict:
    directory = Path("geometry") / new_id()
    retained = []
    try:
        inputs = {
            "reference": store.copy_artifact(first, str(directory / "inputs")),
            "candidate": store.copy_artifact(second, str(directory / "inputs")),
        }
        reference = json.loads(store.resolve(inputs["reference"]["path"]).read_text(encoding="utf-8"))
        candidate = json.loads(store.resolve(inputs["candidate"]["path"]).read_text(encoding="utf-8"))
        result = compare_geometry(reference, candidate, fit)
        result["inputs"] = inputs
        path = store.resolve(str(directory / "report.json"))
        result["report"] = str(path)
        atomic_json(path, result)
        retained = [str(directory / "report.json"), *(artifact["path"] for artifact in inputs.values())]
        return result
    finally:
        cleanup_owned_files(store, str(directory), retained)
