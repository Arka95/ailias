"""Own immutable reel jobs, native worker evidence, audio-solve gates and explicit video publication."""

import json
import math
import re
import subprocess
import uuid
from pathlib import Path

from ailias.generation.tools import NativeTools
from ailias.reanimation.media import encode_command, inspect_audio, inspect_frames, verify_video
from ailias.storage import atomic_json

REQUIRED_PLUGINS = ("PythonScriptPlugin", "EditorScriptingUtilities", "MetaHuman",
                    "MovieRenderPipeline", "SequencerScripting")


def doctor(editor, project, encoder=None):
    blockers = []
    if not editor.is_file():
        blockers.append("Unreal editor executable is missing")
    if not project.is_file() or project.suffix.lower() != ".uproject":
        blockers.append("Existing .uproject is required; no project is silently modified")
    else:
        content = json.loads(project.read_text(encoding="utf-8-sig"))
        enabled = {p["Name"] for p in content.get("Plugins", []) if p.get("Enabled")}
        blockers.extend("Enable project plugin: " + name for name in REQUIRED_PLUGINS if name not in enabled)
    if encoder is not None and not encoder.is_file():
        blockers.append("Explicit FFmpeg executable is missing")
    return {"editor": str(editor.resolve()), "project": str(project.resolve()), "blockers": blockers,
            "required_input": "Assembled rigged MetaHuman Blueprint with Face skeletal component and RigLogic",
            "static_head_or_character_draft_is_sufficient": False,
            "render_backend": "D3D12 GPU; no NullRHI or CPU render fallback",
            "solver_backend": "Installed Speech2Face uses CPU NNERuntimeORTCpu, not CUDA",
            "encoder": str(encoder.resolve()) if encoder else None}


def inspect_job(store, job):
    if not re.fullmatch(r"[a-f0-9]{32}", job):
        raise ValueError("Reel job ID must be the generated 32-character ID")
    path = store.resolve(str(Path("reanimation") / "runs" / job / "manifest.json"))
    report = json.loads(path.read_text())
    report["integrity_problems"] = [name for name, ref in report.get("artifacts", {}).items()
                                    if not store.verify(ref)]
    return report


def create_reel(store, audio, editor, project, policy, *, blueprint=None, encoder=None, tools=None):
    if blueprint and not re.fullmatch(r"/Game/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)?", blueprint):
        raise ValueError("MetaHuman Blueprint must be a /Game asset path")
    source_audio = inspect_audio(audio, policy.max_audio_seconds)
    readiness = doctor(editor, project, encoder)
    if blueprint and encoder is None:
        readiness["blockers"].append("Rendering requires --ffmpeg for final H.264/AAC video")
    revision = uuid.uuid4().hex
    root = store.resolve(str(Path("reanimation") / "runs" / revision))
    root.mkdir(parents=True, exist_ok=False)
    report = {"revision": revision, "status": "blocked" if readiness["blockers"] else "running",
              "readiness": readiness, "audio": source_audio, "policy": policy.to_dict(),
              "framing": policy.framing(), "artifacts": {},
              "blueprint": blueprint, "video_verified": False, "likeness_verified": False}
    report["artifacts"]["audio"] = store.reference_artifact(audio)
    manifest = root / "manifest.json"
    atomic_json(manifest, report)
    if readiness["blockers"]:
        return report
    request_path, receipt = root / "request.json", root / "native-receipt.json"
    frames = root / "frames"
    frames.mkdir()
    frame_count = math.ceil(source_audio["duration_seconds"]*policy.fps)
    request = {"destination": f"/Game/Ailias/Reanimation/R{revision}",
               "audio": str(audio.resolve()), "receipt": str(receipt), "blueprint": blueprint,
               "policy": policy.to_dict(), "framing": policy.framing(),
               "frames": str(frames), "frame_count": frame_count}
    atomic_json(request_path, request)
    report["artifacts"]["worker"] = store.copy_artifact(
        Path(__file__).with_name("unreal_worker.py"), str(root.relative_to(store.root) / "worker"))
    native = tools or NativeTools(editor.parent, policy.timeout_seconds)
    native.paths["unreal"] = editor.resolve()
    try:
        report["runtime"] = native.run("unreal", "reanimation", [
            str(project.resolve()), "-ExecutePythonScript=" + str(store.artifact_path(report["artifacts"]["worker"])),
            "-unattended", "-nop4", "-nosplash", "-AllowCommandletRendering",
            "-RenderOffscreen", "-d3d12", "-stdout", "-FullStdOutLogOutput",
            "-ExecCmds=r.DefaultFeature.AutoExposure 0",
        ], root, environment={"AILIAS_REANIMATION_REQUEST": str(request_path)})
        result = json.loads(receipt.read_text())
        report["native"] = result
        expected_status = "frames_rendered" if blueprint else "audio_solved_awaiting_rigged_metahuman"
        if result.get("status") != expected_status:
            raise ValueError("Native reanimation failed: " + str(result.get("error", result.get("stage"))))
        if result.get("processed_frames", 0) <= 0:
            raise ValueError("Native audio solver did not confirm any processed animation frames")
        assets = ("performance", "sound", "sequence", "map", "material") if blueprint else ("performance", "sound")
        for name in assets:
            asset = result[name].split(".")[0]
            if not asset.startswith(request["destination"] + "/") or ".." in asset:
                raise ValueError("Native output escaped its unique job namespace")
            package = project.parent / "Content" / Path(*asset.removeprefix("/Game/").split("/"))
            suffix = ".umap" if name == "map" else ".uasset"
            report["artifacts"][name] = store.reference_artifact(package.with_suffix(suffix))
        if not blueprint:
            report["status"] = "audio_solved_awaiting_rigged_metahuman"
        else:
            if not result.get("render_success") or result.get("processed_frames", 0) <= 0:
                raise ValueError("Native render did not confirm an animated frame range")
            images = inspect_frames(frames, policy, frame_count)
            for name, image in (("first_frame", images[0]), ("last_frame", images[-1])):
                report["artifacts"][name] = store.reference_artifact(image)
            output = root / "portrait-reel.mp4"
            command = encode_command(encoder, frames, audio, output, policy, frame_count)
            with (root / "encode.log").open("wb") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=policy.timeout_seconds, check=False)
            if process.returncode:
                raise ValueError("Video encoding failed; no CPU encoder fallback; inspect encode.log")
            report["video"] = verify_video(encoder, output, policy, frame_count)
            report["artifacts"]["video"] = store.reference_artifact(output)
            report.update(status="video_rendered_awaiting_visual_review", video_verified=True)
        if not store.verify(report["artifacts"]["audio"]):
            raise ValueError("Source narration changed during animation")
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        report.update(status="failed", video_verified=False,
                      error={"type": type(error).__name__, "message": str(error)})
    finally:
        for path in (request_path, receipt):
            if path.is_file() and path.stat().st_size:
                report["artifacts"][path.name] = store.reference_artifact(path)
        atomic_json(manifest, report)
    return report
