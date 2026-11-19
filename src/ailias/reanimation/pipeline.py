"""Own immutable reel jobs, native worker evidence, audio-solve gates and explicit video publication."""

import json
import math
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ailias.common.code_assets import module_source
from ailias.common.execution import (
    ExecutionChain,
    PipelineStep,
    StageJournal,
    emit_stage_progress,
)
from ailias.generation.tools import NativeTools
from ailias.reanimation.media import (
    encode_command,
    inspect_audio,
    inspect_frames,
    verify_video,
)
from ailias.resource_management import finalize_job_resources
from ailias.storage import atomic_json

REQUIRED_PLUGINS = (
    "PythonScriptPlugin",
    "EditorScriptingUtilities",
    "MetaHuman",
    "MovieRenderPipeline",
    "SequencerScripting",
)


def doctor(editor, project, encoder=None, *, direct_video=False):
    blockers = []
    if not editor.is_file():
        blockers.append("Unreal editor executable is missing")
    if not project.is_file() or project.suffix.lower() != ".uproject":
        blockers.append(
            "Existing .uproject is required; no project is silently modified"
        )
    else:
        content = json.loads(project.read_text(encoding="utf-8-sig"))
        enabled = {p["Name"] for p in content.get("Plugins", []) if p.get("Enabled")}
        blockers.extend(
            "Enable project plugin: " + name
            for name in REQUIRED_PLUGINS
            if name not in enabled
        )
        if direct_video and "AppleProResMedia" not in enabled:
            blockers.append(
                "Enable project plugin: AppleProResMedia, then restart the owned editor"
            )
    if encoder is not None and not encoder.is_file():
        blockers.append("Explicit FFmpeg executable is missing")
    return {
        "editor": str(editor.resolve()),
        "project": str(project.resolve()),
        "blockers": blockers,
        "required_input": "Assembled rigged MetaHuman Blueprint with Face skeletal component and RigLogic",
        "static_head_or_character_draft_is_sufficient": False,
        "render_backend": "D3D12 GPU; no NullRHI or CPU render fallback",
        "solver_backend": "Installed Speech2Face uses CPU NNERuntimeORTCpu, not CUDA",
        "encoder": str(encoder.resolve()) if encoder else None,
    }


def inspect_job(store, job):
    if not re.fullmatch(r"[a-f0-9]{32}", job):
        raise ValueError("Reel job ID must be the generated 32-character ID")
    path = store.resolve(str(Path("reanimation") / "runs" / job / "manifest.json"))
    report = json.loads(path.read_text())
    report["integrity_problems"] = [
        name
        for name, ref in report.get("artifacts", {}).items()
        if not store.verify(ref)
    ]
    return report


def create_reel(
    store,
    audio,
    editor,
    project,
    policy,
    *,
    blueprint=None,
    encoder=None,
    tools=None,
    keep_frames=False,
):
    return ReelPipeline(
        store,
        audio,
        editor,
        project,
        policy,
        blueprint=blueprint,
        encoder=encoder,
        tools=tools,
        keep_frames=keep_frames,
    ).run()


class ReelPipeline:
    """Own one reel job; native work and media publication have separate timing boundaries."""

    def __init__(
        self,
        store,
        audio,
        editor,
        project,
        policy,
        *,
        blueprint=None,
        encoder=None,
        tools=None,
        keep_frames=False,
    ):
        if blueprint and not re.fullmatch(
            r"/Game/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)?", blueprint
        ):
            raise ValueError("MetaHuman Blueprint must be a /Game asset path")
        self.store, self.audio, self.editor, self.project = (
            store,
            audio,
            editor,
            project,
        )
        self.policy, self.blueprint, self.encoder = policy, blueprint, encoder
        self.keep_frames = keep_frames
        self.report = {"started_at": datetime.now(timezone.utc).isoformat()}
        self.journal = StageJournal(self.report, on_progress=emit_stage_progress)
        source_audio = self.journal.execute(
            "validate_audio", lambda: inspect_audio(audio, policy.max_audio_seconds)
        )
        readiness = self.journal.execute(
            "validate_prerequisites",
            lambda: doctor(
                editor,
                project,
                encoder,
                direct_video=bool(blueprint and not keep_frames),
            ),
        )
        if blueprint and encoder is None:
            readiness["blockers"].append(
                "Rendering requires --ffmpeg for final H.264/AAC video"
            )
        revision = uuid.uuid4().hex
        self.root = store.resolve(str(Path("reanimation") / "runs" / revision))
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = self.root / "manifest.json"
        self.journal.manifest = self.manifest
        self.report.update(
            revision=revision,
            status="blocked" if readiness["blockers"] else "running",
            readiness=readiness,
            audio=source_audio,
            policy=policy.to_dict(),
            framing=policy.framing(),
            placement=policy.placement(),
            artifacts={},
            blueprint=blueprint,
            video_verified=False,
            likeness_verified=False,
            keep_frames=keep_frames,
        )
        self.report["artifacts"]["audio"] = store.reference_artifact(audio)
        self.request_path, self.receipt = (
            self.root / "request.json",
            self.root / "native-receipt.json",
        )
        self.frames, self.output = self.root / "frames", self.root / "portrait-reel.mp4"
        self.movie = (
            self.root / "portrait-source.mov" if blueprint and not keep_frames else None
        )
        self.frame_count = math.ceil(source_audio["duration_seconds"] * policy.fps)
        self.request = {
            "destination": f"/Game/Ailias/Reanimation/R{revision}",
            "audio": str(audio.resolve()),
            "receipt": str(self.receipt),
            "blueprint": blueprint,
            "policy": policy.to_dict(),
            "framing": policy.framing(),
            "placement": policy.placement(),
            "frames": str(self.frames),
            "frame_count": self.frame_count,
        }
        if self.movie is not None:
            self.request["movie"] = str(self.movie)
        self.report["render_output"] = "prores_movie" if self.movie else "png_sequence"
        self.native = tools or NativeTools(editor.parent, policy.timeout_seconds)
        self.native.paths["unreal"] = editor.resolve()
        self.expected_status = (
            ("movie_rendered" if self.movie else "frames_rendered")
            if blueprint
            else "audio_solved_awaiting_rigged_metahuman"
        )
        self.journal.persist()

    def prepare_request(self):
        if self.movie is None:
            self.frames.mkdir()
        directory = str(self.root.relative_to(self.store.root) / "worker")
        for name, module in (
            ("worker", "ailias.reanimation.unreal_worker"),
            ("stage_journal", "ailias.common.execution"),
        ):
            self.report["artifacts"][name] = self.store.copy_artifact(
                module_source(module), directory
            )
        helper = self.report["artifacts"]["stage_journal"]
        self.request["stage_journal"] = {
            "path": str(self.store.artifact_path(helper)),
            "sha256": helper["sha256"],
        }
        atomic_json(self.request_path, self.request)

    def execute_native(self):
        from ailias.generation.unreal_session import execute_unreal

        self.report["runtime"] = execute_unreal(
            self.native,
            self.project,
            self.store.artifact_path(self.report["artifacts"]["worker"]),
            self.request_path,
            "reanimation",
            self.root,
            "AILIAS_REANIMATION_REQUEST",
            self.expected_status,
            console_commands="r.DefaultFeature.AutoExposure 0",
        )

    def verify_native_outputs(self):
        result = json.loads(self.receipt.read_text())
        self.report["native"] = result
        if result.get("status") != self.expected_status:
            raise ValueError(
                "Native reanimation failed: "
                + str(result.get("error", result.get("stage")))
            )
        if result.get("processed_frames", 0) <= 0:
            raise ValueError(
                "Native audio solver did not confirm any processed animation frames"
            )
        assets = (
            ("performance", "sound", "sequence", "map", "material")
            if self.blueprint
            else ("performance", "sound")
        )
        for name in assets:
            asset = result[name].split(".")[0]
            if not asset.startswith(self.request["destination"] + "/") or ".." in asset:
                raise ValueError("Native output escaped its unique job namespace")
            package = (
                self.project.parent
                / "Content"
                / Path(*asset.removeprefix("/Game/").split("/"))
            )
            suffix = ".umap" if name == "map" else ".uasset"
            self.report["artifacts"][name] = self.store.reference_artifact(
                package.with_suffix(suffix)
            )
        if not self.blueprint:
            self.report["status"] = self.expected_status
        elif not result.get("render_success"):
            raise ValueError("Native render did not confirm an animated frame range")

    def validate_frames(self):
        images = inspect_frames(self.frames, self.policy, self.frame_count)
        for name, image in (("first_frame", images[0]), ("last_frame", images[-1])):
            self.report["artifacts"][name] = self.store.reference_artifact(image)

    def validate_movie(self):
        if not self.movie.is_file() or self.movie.stat().st_size == 0:
            raise ValueError("Unreal did not produce the requested direct movie")

    def encode_video(self):
        command = encode_command(
            self.encoder,
            self.movie if self.movie is not None else self.frames,
            self.audio,
            self.output,
            self.policy,
            self.frame_count,
        )
        with (self.root / "encode.log").open("wb") as stream:
            process = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=self.policy.timeout_seconds,
                check=False,
            )
        if process.returncode:
            raise ValueError(
                "Video encoding failed; no CPU encoder fallback; inspect encode.log"
            )

    def verify_encoded_video(self):
        self.report["video"] = verify_video(
            self.encoder, self.output, self.policy, self.frame_count
        )
        self.report["artifacts"]["video"] = self.store.reference_artifact(self.output)
        self.report.update(
            status="video_rendered_awaiting_visual_review", video_verified=True
        )

    def verify_sources(self):
        if not self.store.verify(self.report["artifacts"]["audio"]):
            raise ValueError("Source narration changed during animation")

    def cleanup(self):
        retained_frames = []
        images = sorted(
            path
            for path in self.frames.glob("frame.*.png")
            if path.is_file() and path.stat().st_size
        )
        if self.keep_frames:
            retained_frames = images
        if images:
            for name, image in (("first_frame", images[0]), ("last_frame", images[-1])):
                self.report["artifacts"].setdefault(
                    name, self.store.reference_artifact(image)
                )
        for path in (self.request_path, self.receipt):
            if path.is_file() and path.stat().st_size:
                self.report["artifacts"][path.name] = self.store.reference_artifact(
                    path
                )
        self.report["retention"] = {
            "policy": "explicit_maintenance",
            "deferred": True,
            "reason": "Alias-wide retention is not part of video generation",
        }
        finalize_job_resources(
            self.store,
            self.report,
            self.manifest,
            retained_frames,
            automatic_retention=False,
        )

    def run(self):
        failure = None
        try:
            if self.report["status"] == "blocked":
                return self.report
            steps = [
                PipelineStep("prepare_request", self.prepare_request),
                PipelineStep("native_execution", self.execute_native),
                PipelineStep("verify_native_outputs", self.verify_native_outputs),
            ]
            if self.blueprint:
                steps.extend(
                    [
                        PipelineStep("validate_movie", self.validate_movie)
                        if self.movie is not None
                        else PipelineStep("validate_frames", self.validate_frames),
                        PipelineStep("encode_video", self.encode_video),
                        PipelineStep("verify_encoded_video", self.verify_encoded_video),
                    ]
                )
            steps.append(PipelineStep("verify_sources", self.verify_sources))
            ExecutionChain(steps).run(self.journal)
        except BaseException as error:
            failure = error
            self.report.update(
                status="failed" if isinstance(error, Exception) else "interrupted",
                video_verified=False,
                error={"type": type(error).__name__, "message": str(error)},
            )
            if self.receipt.is_file():
                try:
                    self.report["native"] = json.loads(self.receipt.read_text())
                except (OSError, ValueError):
                    pass
            if not isinstance(error, Exception):
                raise
        finally:
            if self.report["status"] != "blocked":
                try:
                    self.journal.execute("finalize_resources", self.cleanup)
                except Exception as error:
                    self.report["cleanup_error_type"] = type(error).__name__
                    if failure is None:
                        failure = error
                        self.report.update(
                            status="failed",
                            video_verified=False,
                            error={"type": type(error).__name__, "message": str(error)},
                        )
            self.report.update(
                seconds=self.journal.seconds,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            self.journal.persist(failure)
        return self.report
