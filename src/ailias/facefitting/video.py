"""Extract timestamped local video evidence without treating expressions as identity."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import uuid
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation

from ailias.imageprocessing.dense import DenseDetector
from ailias.facefitting.model import model_assets
from ailias.settings import PROJECT_ROOT
from ailias.storage import atomic_json


DEFAULT_VIDEO_CONFIG = PROJECT_ROOT / "config" / "video-evidence.json"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
EXPRESSION_PREFIXES = ("mouth", "brow", "cheek", "noseSneer", "eyeWide", "eyeSquint")


@dataclass(frozen=True)
class VideoPolicy:
    frame_mode: str = "sampled"
    sample_fps: float = 1
    max_samples_per_video: int = 5000
    max_duration_seconds: float = 120
    image_long_side: int = 1920
    detection_long_side: int = 1280
    max_identity_frames: int = 24
    minimum_spacing_seconds: float = 1
    validation_guard_seconds: float = 2
    neutral_expression_max: float = .18
    jaw_open_max: float = .1
    blink_max: float = .2
    minimum_eye_span_pixels: float = 100
    minimum_laplacian_variance: float = 20

    def __post_init__(self):
        if not isinstance(self.frame_mode, str) or self.frame_mode not in {"all", "sampled"}:
            raise ValueError("frame_mode must be all or sampled")
        bounds = {
            "sample_fps": (.1, 10), "max_samples_per_video": (1, 20000),
            "max_duration_seconds": (1, 600), "image_long_side": (256, 8192),
            "detection_long_side": (256, 2048),
            "max_identity_frames": (3, 100), "minimum_spacing_seconds": (.1, 10),
            "validation_guard_seconds": (.1, 30), "neutral_expression_max": (0, 1),
            "jaw_open_max": (0, 1), "blink_max": (0, 1),
            "minimum_eye_span_pixels": (10, 2000), "minimum_laplacian_variance": (0, 10000),
        }
        for key, (lower, upper) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite numeric")
            if not lower <= value <= upper:
                raise ValueError(f"{key} outside supported range")
        for key in ("max_samples_per_video", "image_long_side", "detection_long_side", "max_identity_frames"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")


def frame_evidence(rgb, detection, policy):
    """Keep source-pixel coordinates; pose/depth/expression remain model predictions."""
    import cv2

    points = np.asarray(detection["faces"][0], dtype=float) if len(detection["faces"]) == 1 else None
    if points is None or points.shape != (478, 3) or not np.isfinite(points).all():
        return {"eligible_identity": False, "reasons": ["invalid_or_multiple_faces"]}
    height, width = rgb.shape[:2]
    xy = points[:, :2]*[width, height]
    inside = ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all(1)
    span = float(np.linalg.norm(xy[33]-xy[263]))
    low, high = xy[:468].min(0), xy[:468].max(0)
    x0, y0 = np.clip(np.floor(low).astype(int), 0, [width-1, height-1])
    x1, y1 = np.clip(np.ceil(high).astype(int), 1, [width, height])
    if x1-x0 < 3 or y1-y0 < 3:
        return {"eligible_identity": False, "reasons": ["invalid_face_bounds"]}
    crop = Image.fromarray(rgb).crop((x0, y0, x1, y1)).convert("L")
    crop.thumbnail((256, 256))
    gray = np.asarray(crop)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blends = detection["blendshapes"][0] if detection["blendshapes"] else {}
    expression = max((value for key, value in blends.items()
                      if key.startswith(EXPRESSION_PREFIXES)), default=1.)
    jaw = float(blends.get("jawOpen", 1.))
    blink = max(blends.get("eyeBlinkLeft", 1.), blends.get("eyeBlinkRight", 1.))
    transform = np.asarray(detection["transforms"][0], dtype=float) if detection["transforms"] else None
    if (transform is None or transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=.02)):
        return {"eligible_identity": False, "reasons": ["invalid_pose_transform"]}
    pitch, yaw, roll = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    category = "front" if abs(yaw) < 15 else ("negative" if yaw < 0 else "positive") + (
        "_oblique" if abs(yaw) < 50 else "_profile")
    reasons = []
    if span < policy.minimum_eye_span_pixels:
        reasons.append("face_too_small")
    if sharpness < policy.minimum_laplacian_variance:
        reasons.append("blur_or_low_detail")
    if not inside[:468].all():
        reasons.append("face_clipped")
    if not blends:
        reasons.append("missing_expression_evidence")
    if expression > policy.neutral_expression_max or jaw > policy.jaw_open_max:
        reasons.append("non_neutral_expression")
    if blink > policy.blink_max:
        reasons.append("blink")
    if abs(pitch) > 25 or abs(yaw) > 70 or abs(roll) > 25:
        reasons.append("extreme_pose")
    dark, bright = float((gray < 12).mean()), float((gray > 245).mean())
    if dark > .3 or bright > .2:
        reasons.append("poor_exposure")
    return {
        "source_size": [width, height], "source_xy": xy.tolist(),
        "on_head": inside.tolist(), "eye_span_source": span,
        "landmark_depth_is_measured": False, "per_point_visibility_available": False,
        "blendshapes": blends, "facial_transform": transform.tolist(),
        "pose": {"yaw_deg": float(yaw), "pitch_deg": float(pitch),
                 "roll_deg": float(roll), "category": category, "calibrated": False},
        "face_box": [int(x0), int(y0), int(x1), int(y1)],
        "quality": {"laplacian_variance": sharpness, "dark_fraction": dark,
                    "bright_fraction": bright, "expression_max": float(expression),
                    "jaw_open": jaw, "blink_max": float(blink)},
        "eligible_identity": not reasons, "reasons": reasons,
        "quality_score": math.log1p(sharpness)-4*expression-4*jaw-2*blink,
    }


def select_identity(frames, policy):
    """Balance pose bins and reserve later time blocks with a no-neighbour guard."""
    groups = {}
    for frame in frames:
        if frame.get("eligible_identity"):
            groups.setdefault(frame["pose"]["category"], []).append(frame)
    for group in groups.values():
        group.sort(key=lambda f: (-f["quality_score"], f["face_id"]))
    chosen = []
    while groups and len(chosen) < policy.max_identity_frames:
        for category in list(groups):
            group = groups[category]
            accepted = None
            while group:
                frame = group.pop(0)
                if all(frame["source_id"] != old["source_id"] or
                       abs(frame["timestamp_seconds"]-old["timestamp_seconds"])
                       >= policy.minimum_spacing_seconds for old in chosen):
                    accepted = frame
                    break
            if accepted:
                chosen.append(accepted)
            if not group:
                del groups[category]
            if len(chosen) == policy.max_identity_frames:
                break
    validation = []
    for category in sorted({f["pose"]["category"] for f in chosen}):
        group = [f for f in chosen if f["pose"]["category"] == category]
        if len(group) < 2:
            continue
        # A temporal block, not random adjacent frames, is held out.
        holdout = max(group, key=lambda f: (f["timestamp_seconds"], f["face_id"]))
        independent = [f for f in group if f["source_id"] != holdout["source_id"] or
                       abs(f["timestamp_seconds"]-holdout["timestamp_seconds"])
                       >= policy.validation_guard_seconds]
        if independent:
            validation.append(holdout)
    training = [frame for frame in chosen if frame not in validation and all(
        frame["source_id"] != other["source_id"] or
        abs(frame["timestamp_seconds"]-other["timestamp_seconds"]) >= policy.validation_guard_seconds
        for other in validation)]
    return {"training": [f["face_id"] for f in training],
            "validation": [f["face_id"] for f in validation],
            "pose_categories": sorted({f["pose"]["category"] for f in chosen}),
            "validation_independent_capture": False}


def select_expression_aware(frames, policy):
    """Keep the neutral gate strict; use a different contract for a joint-expression fit."""
    candidates = []
    for frame in frames:
        if (set(frame["reasons"]) <= {"non_neutral_expression"}
                and frame.get("quality", {}).get("jaw_open", 1) <= .35):
            candidates.append({**frame, "eligible_identity": True})
    result = select_identity(candidates, policy)
    return {**result, "method": "shared_identity_with_per_frame_camera_and_expression",
            "frames_are_neutral": False, "expression_model_required": True}


def sampled_frames(path, policy):
    """Decode sequentially using presentation timestamps, never index / average FPS."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Input has no video stream")
        stream = container.streams.video[0]
        first, previous, next_time, count = None, None, 0., 0
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video frame lacks a presentation timestamp")
            absolute = float(frame.pts*frame.time_base)
            if previous is not None and absolute <= previous:
                raise ValueError("Non-monotonic video presentation timestamps")
            previous = absolute
            if first is None:
                first = absolute
            relative = absolute-first
            if relative > policy.max_duration_seconds or count >= policy.max_samples_per_video:
                raise ValueError("Video exceeds the configured decoding budget; no complete timeline claimed")
            if policy.frame_mode == "sampled" and relative+1e-8 < next_time:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            rotation = float(getattr(frame, "rotation", 0) or stream.metadata.get("rotate", 0))
            if rotation % 360:
                raise ValueError("Rotated video requires an explicitly normalized source copy")
            yield rgb, {"decode_index": index, "pts": frame.pts, "time_base": str(frame.time_base),
                        "timestamp_seconds": relative, "container_time_seconds": absolute,
                        "video_start_seconds": first}
            count += 1
            next_time = (math.floor(relative*policy.sample_fps+1e-6)+1)/policy.sample_fps
        if not count:
            raise ValueError("Video decoded no frames")
        if policy.frame_mode == "all" and stream.frames and count != stream.frames:
            raise ValueError("Decoded video frame count differs from the original stream")


def extract_audio(path, output):
    """Decode PCM16 with a retained clock origin; reject gaps rather than invent sync."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return {"status": "no_audio_stream"}
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        first, samples = None, 0
        with wave.open(str(output), "wb") as target:
            target.setparams((1, 2, 48000, 0, "NONE", "not compressed"))

            def write(frames):
                nonlocal first, samples
                for frame in frames:
                    if frame.pts is None:
                        raise ValueError("Audio lacks timing evidence")
                    timestamp = float(frame.pts*frame.time_base)
                    if first is None:
                        first = timestamp
                    if abs(timestamp-first-samples/48000) > 2/48000:
                        raise ValueError("Audio timestamp gap prevents exact WAV/video alignment")
                    target.writeframes(frame.to_ndarray().tobytes())
                    samples += frame.samples

            for frame in container.decode(stream):
                write(resampler.resample(frame))
            write(resampler.resample(None))
        if not samples:
            raise ValueError("Audio stream decoded no samples")
    return {"status": "decoded", "sample_rate": 48000, "samples": samples,
            "channels": 1, "sample_width": 2, "audio_start_seconds": first,
            "duration_seconds": samples/48000}


def contact_sheet(store, root, frames, name):
    width, height, columns = 240, 280, 6
    if not frames:
        return None
    sheet = Image.new("RGB", (columns*width, math.ceil(len(frames)/columns)*height), "#181818")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        with Image.open(store.artifact_path(frame["image"])) as image:
            box = frame.get("face_box", (0, 0, image.width, image.height))
            margin = (box[2]-box[0])*.25
            crop = image.crop((max(0, box[0]-margin), max(0, box[1]-margin),
                               min(image.width, box[2]+margin), min(image.height, box[3]+margin)))
            crop = ImageOps.contain(crop.convert("RGB"), (width, height-45))
        x, y = index % columns*width, index//columns*height
        sheet.paste(crop, (x, y+45))
        draw.text((x+4, y+3), f"{frame['source_name']} {frame['timestamp_seconds']:.2f}s", fill="white")
        draw.text((x+4, y+20), frame.get("pose", {}).get("category", "rejected"), fill="white")
    path = root / name
    sheet.save(path, quality=94)
    return store.reference_artifact(path)


def extract_evidence(store, folder, policy, *, detector=None):
    """Own only a fresh UUID revision; no shared session-state writes or source edits."""
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if not paths:
        raise ValueError("No supported video files in input folder")
    root = store.resolve(str(Path("facefitting") / "video" / uuid.uuid4().hex))
    root.mkdir(parents=True, exist_ok=False)
    report = {"revision": root.name, "status": "running", "policy": asdict(policy),
              "sources": {}, "frames": [], "artifacts": {}, "geometry_generated": False,
              "limitations": [
                  "Monocular RGB, not measured 3D/depth; pose and neutrality are predictions.",
                  "Shared clothing/background do not supply metric camera or light calibration.",
                  "Identity uses relaxed frames; expressions and audio are separate performance evidence.",
                  "Validation has temporal separation, not independent identity ground truth.",
                  "No de-lighting, geometry fitting, Blender, cloud upload or rig request in extraction.",
              ]}
    manifest = root / "manifest.json"
    atomic_json(manifest, report)
    owns_detector = detector is None
    try:
        if owns_detector:
            asset = model_assets(store)["landmarker"]["artifact"]
            report["landmarker"] = asset
            detector = DenseDetector(store.artifact_path(asset))
        for path in paths:
            source = store.reference_artifact(path)
            source_id = source["sha256"]
            if source_id in report["sources"]:
                continue
            video_root = root / source_id[:16]
            video_root.mkdir()
            source_report = {"source": source, "name": path.name}
            report["sources"][source_id] = source_report
            start_count = len(report["frames"])
            for rgb, timing in sampled_frames(path, policy):
                image = Image.fromarray(rgb)
                original_size = list(image.size)
                image.thumbnail((policy.image_long_side, policy.image_long_side), Image.Resampling.LANCZOS)
                small = image.copy()
                small.thumbnail((policy.detection_long_side, policy.detection_long_side))
                detection = detector.detect(np.asarray(small))
                evidence = frame_evidence(np.asarray(image), detection, policy)
                face_id = f"{source_id[:16]}-{timing['pts']}"
                target = video_root / f"{face_id}.jpg"
                image.save(target, quality=95, subsampling=0)
                frame = {**evidence, **timing, "face_id": face_id, "source_id": source_id,
                         "source_name": path.name, "original_source_size": original_size,
                         "image": store.reference_artifact(target)}
                report["frames"].append(frame)
                report["artifacts"][face_id] = frame["image"]
            source_report["decoded_images"] = len(report["frames"])-start_count
            source_report["all_frames_retained"] = policy.frame_mode == "all"
            source_report["audio"] = extract_audio(path, video_root / "audio.wav")
            if source_report["audio"]["status"] == "decoded":
                reference = store.reference_artifact(video_root / "audio.wav")
                source_report["audio"]["artifact"] = reference
                report["artifacts"][source_id+"-audio"] = reference
            if not store.verify(source):
                raise ValueError("Source recording changed during extraction")
            atomic_json(manifest, report)
        report["selection"] = select_identity(report["frames"], policy)
        report["fitting_selection"] = select_expression_aware(report["frames"], policy)
        ids = report["fitting_selection"]["training"]+report["fitting_selection"]["validation"]
        selected = [f for f in report["frames"] if f["face_id"] in ids]
        preview = contact_sheet(store, root, selected, "identity-contact.jpg")
        if preview:
            report["artifacts"]["identity_contact"] = preview
        expression_frames = sorted(
            (f for f in report["frames"] if "quality" in f),
            key=lambda f: -f["quality"]["expression_max"])[::max(1, len(report["frames"])//24)][:24]
        preview = contact_sheet(store, root, expression_frames, "expression-contact.jpg")
        if preview:
            report["artifacts"]["expression_contact"] = preview
        report["status"] = ("video_evidence_ready" if len(report["fitting_selection"]["training"]) >= 3
                            and report["fitting_selection"]["validation"] else "insufficient_fitting_coverage")
        if not all(store.verify(s["source"]) for s in report["sources"].values()):
            raise ValueError("Input recording changed")
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        try:
            if owns_detector and detector is not None:
                detector.close()
        finally:
            atomic_json(manifest, report)
    return report
