"""Compare generic photo/render observations before changing a native Character."""

import json
import uuid
from pathlib import Path

from ailias.facefitting.native_features_worker import validate_profile
from ailias.facefitting.native_reference import iris_tint_adjustment
from ailias.storage import atomic_json


def compare_features(store, reference_path, reference_face_id, candidate_path, current_tint=None):
    """Publish diagnostics and an optional bounded eye proposal, never a likeness pass."""
    dependencies = []
    profiles = []
    for path in (reference_path, candidate_path):
        artifact = store.reference_artifact(path)
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
        if (profile.get("schema_version") != 1
                or profile.get("method") != "photo-regions-and-projected-landmarks-v1"
                or profile.get("status") != "features_extracted_awaiting_review"):
            raise ValueError("Comparison requires complete versioned photo-feature evidence")
        dependencies.extend([artifact, *profile["input_dependencies"],
                             *profile["artifacts"].values()])
        profiles.append(profile)
    reference, candidate = profiles
    if reference_face_id not in reference["faces"]:
        raise ValueError("Unknown reference face ID")
    if candidate.get("image_role") != "render" or len(candidate["faces"]) != 1:
        raise ValueError("Candidate evidence must explicitly describe one rendered face")
    first, second = reference["faces"][reference_face_id], next(iter(candidate["faces"].values()))
    if first["status"] != "observed" or second["status"] != "observed":
        raise ValueError("Comparison requires observed faces")
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Photo/render comparison evidence changed")
    a, b = first["features"], second["features"]
    geometry, omitted = {}, []
    for name, measurement in a["geometry_2d"].items():
        other = b["geometry_2d"].get(name, {})
        if measurement["status"] != "observed" or other.get("status") != "observed":
            omitted.append(name)
            continue
        reference_ratio, candidate_ratio = measurement["outer_eye_span_ratio"], other["outer_eye_span_ratio"]
        geometry[name] = {
            "reference_ratio": reference_ratio, "candidate_ratio": candidate_ratio,
            "signed_delta": candidate_ratio-reference_ratio,
        }
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("facefitting") / "feature-comparisons" / revision))
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "revision": revision, "status": "comparison_requires_visual_review",
        "input_dependencies": dependencies, "reference_face_id": reference_face_id,
        "candidate_face_id": next(iter(candidate["faces"])),
        "reference_cohort": reference["cohort"], "candidate_cohort": candidate["cohort"],
        "geometry_2d": geometry, "unmeasurable_geometry": omitted,
        "expression": {"reference": first.get("expression", {}), "candidate": second.get("expression", {})},
        "artifacts": {}, "likeness_verified": False, "native_edits_applied": False,
        "limitations": [
            "Projected ratios depend on camera/pose/expression; differences are not isolated 3D shape errors",
            "A smiling reference and neutral candidate are not expression-matched",
            "Photo/render colours retain different lighting; a tint proposal still needs native render verification",
            "No automatic pass threshold, single likeness score or fabricated millimetre accuracy",
        ],
    }
    if current_tint is not None:
        if a["iris"]["status"] != "observed" or b["iris"]["status"] != "observed":
            report["iris_proposal"] = {"status": "review", "reason": "insufficient_iris_evidence"}
        else:
            adjustment = iris_tint_adjustment(a["iris"], b["iris"], current_tint)
            report["iris_proposal"] = adjustment
            profile = validate_profile({
                "version": 1, "eyes": adjustment["eyes"], "sources": dependencies,
            })
            profile_path = root / "iris-proposal.json"
            atomic_json(profile_path, profile)
            report["artifacts"]["iris_proposal"] = store.reference_artifact(profile_path)
    if not all(store.verify(item) for item in dependencies):
        raise ValueError("Photo/render comparison evidence changed while comparing")
    path = root / "manifest.json"
    report["manifest"] = str(path)
    atomic_json(path, report)
    return report
