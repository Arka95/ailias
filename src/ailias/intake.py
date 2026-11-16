"""Reference input photos, assess image quality and record human capture reviews.

Coverage validation uses reviewed capture evidence, not automatic face matches.
"""

import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from ailias.settings import Policy
from ailias.storage import Store, invalidate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
VIEWS = {
    "front": (0, 0),
    "left_30": (-30, 0),
    "right_30": (30, 0),
    "left_45": (-45, 0),
    "right_45": (45, 0),
    "left_profile": (-90, 0),
    "right_profile": (90, 0),
    "above": (0, 20),
    "below": (0, -20),
}


def analyze_image(path: Path, policy: Policy) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            width, height = image.size
            gray = image.convert("L")
            gray.thumbnail((512, 512))
            histogram = gray.histogram()
            count = gray.width * gray.height
            dark = sum(histogram[:16]) / count
            bright = sum(histogram[240:]) / count
            energy = 0.0
            if min(gray.size) > 2:
                edges = gray.filter(ImageFilter.FIND_EDGES)
                edges = edges.crop((1, 1, gray.width - 1, gray.height - 1))
                energy = sum(i * i * n for i, n in enumerate(edges.histogram()))
                energy /= edges.width * edges.height
            reasons = []
            if min(width, height) < policy.min_short_side:
                reasons.append("resolution_below_minimum")
            if energy < policy.min_edge_energy:
                reasons.append("low_detail_or_blur")
            if dark > policy.max_dark_fraction:
                reasons.append("underexposed")
            if bright > policy.max_bright_fraction:
                reasons.append("overexposed")
            return {
                "width": width, "height": height,
                "edge_energy": round(energy, 4),
                "dark_fraction": round(dark, 4),
                "bright_fraction": round(bright, 4),
                "reasons": reasons,
            }


def ingest(store: Store, folder: Path, policy: Policy) -> dict:
    if folder.is_file() and folder.name == "selection.json":
        selection = json.loads(folder.read_text(encoding="utf-8"))
        if selection.get("alias") != store.root.name:
            raise ValueError("Selection belongs to a different alias")
        artifacts = [member["products"]["portrait"] for member in selection["members"]]
        if not all(store.verify(artifact) for artifact in artifacts):
            raise ValueError("Selected portrait is missing or changed")
        files = sorted({store.artifact_path(artifact) for artifact in artifacts})
    elif folder.is_dir():
        files = sorted(path for path in folder.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise ValueError("Photo folder or selection.json does not exist")
    if not files:
        raise ValueError("No supported images found")
    state = store.load()
    outcomes = []
    changed = False
    for path in files:
        try:
            quality = analyze_image(path, policy)
            artifact = store.reference_artifact(path)
        except (OSError, ValueError, UnidentifiedImageError,
                Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
            outcomes.append({"name": path.name, "error": type(error).__name__})
            continue
        photo_id = artifact["sha256"]
        record = {"artifact": artifact, "quality": quality, "review": None}
        previous = state["photos"].get(photo_id)
        if previous:
            record["review"] = previous["review"]
            if store.verify(previous["artifact"]):
                record["artifact"] = previous["artifact"]
        if previous != record:
            state["photos"][photo_id] = record
            changed = True
        outcomes.append({"name": path.name, "id": photo_id, "quality": quality})
    if changed:
        invalidate(state, "intake", "Photo evidence changed")
        store.save(state)
    return {"photos": outcomes, "total": len(state["photos"])}


def review_photo(store: Store, photo_id: str, yaw: float, pitch: float,
                 neutral: bool, unobstructed: bool, consistent: bool,
                 landmarks: dict | None = None) -> dict:
    if not math.isfinite(yaw) or not -180 <= yaw <= 180:
        raise ValueError("Yaw must be finite and between -180 and 180")
    if not math.isfinite(pitch) or not -90 <= pitch <= 90:
        raise ValueError("Pitch must be finite and between -90 and 90")
    if landmarks is not None:
        if not isinstance(landmarks, dict) or not landmarks:
            raise ValueError("Landmarks must be a non-empty object of named normalized [x,y] pairs")
        for point in landmarks.values():
            if (not isinstance(point, list) or len(point) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or not 0 <= v <= 1 for v in point)):
                raise ValueError("Landmark coordinates must be finite numbers between 0 and 1")
    state = store.load()
    matches = [key for key in state["photos"] if key.startswith(photo_id)]
    if not photo_id or len(matches) != 1:
        raise ValueError("Photo ID prefix must match exactly one photo")
    review = {
        "source": "manual", "yaw": yaw, "pitch": pitch,
        "neutral": neutral, "unobstructed": unobstructed,
        "consistent_capture": consistent, "landmarks": landmarks,
    }
    photo = state["photos"][matches[0]]
    if photo["review"] != review:
        photo["review"] = review
        invalidate(state, "intake", "Manual photo review changed")
        store.save(state)
    return review


def validate(store: Store, policy: Policy) -> dict:
    state = store.load()
    accepted, rejected, covered = [], {}, set()
    for photo_id, photo in state["photos"].items():
        path = store.artifact_path(photo["artifact"])
        reasons = []
        if not store.verify(photo["artifact"]):
            reasons.append("source_missing_or_changed")
        else:
            reasons.extend(analyze_image(path, policy)["reasons"])
        review = photo["review"]
        if review is None:
            reasons.append("manual_review_required")
        else:
            for key, reason in (
                ("neutral", "non_neutral_expression"),
                ("unobstructed", "face_obstructed"),
                ("consistent_capture", "inconsistent_lighting_or_focal_length"),
            ):
                if not review[key]:
                    reasons.append(reason)
        if reasons:
            rejected[photo_id] = reasons
            continue
        accepted.append(photo_id)
        nearest = min(VIEWS, key=lambda view: (
            (review["yaw"] - VIEWS[view][0]) ** 2
            + (review["pitch"] - VIEWS[view][1]) ** 2
        ))
        yaw, pitch = VIEWS[nearest]
        if (abs(review["yaw"] - yaw) <= policy.angle_tolerance
                and abs(review["pitch"] - pitch) <= policy.angle_tolerance):
            covered.add(nearest)
    missing = [view for view in VIEWS if view not in covered]
    ready = policy.min_photos <= len(accepted) <= policy.max_photos and not missing
    report = {
        "ready": ready, "accepted": accepted, "rejected": rejected,
        "accepted_count": len(accepted),
        "count_range": [policy.min_photos, policy.max_photos],
        "missing_views": missing,
        "requests": [
            {"view": view, "yaw": VIEWS[view][0], "pitch": VIEWS[view][1],
             "instruction": "Keep neutral expression, diffuse lighting and fixed focal length"}
            for view in missing
        ],
        "angle_source": "manual review; no pose or expression model has run",
        "quality_method": "uncalibrated whole-image edge-energy and exposure heuristics",
        "policy": asdict(policy),
    }
    previous = state["stages"]["intake"]
    new = {"status": "complete" if ready else "blocked", "report": report}
    if previous != new:
        invalidate(state, "intake", "Photo validation or policy changed")
        state["stages"]["intake"] = new
        store.save(state)
    return report
