"""Fresh sampled-photo identity fitting and native Character creation, without prior heads."""

from contextlib import redirect_stdout
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from ailias.common.code_assets import module_source
import sys
import time
import uuid
import numpy as np
from PIL import Image
from ailias.metahuman.appearance_mapping import build_appearance_profile
from ailias.head_reconstruction.blender.appearance_scene import create_blender_head
from ailias.head_reconstruction.appearance.skin_texture_projection import (
    bake_photo_skin_texture,
    estimate_photo_alignment,
)
from ailias.head_reconstruction.settings import VideoPolicy
from ailias.pipelines.settings import DEFAULT_REVIEW_CONFIG, load_appearance_policy
from ailias.head_reconstruction.evidence.photo_processing import (
    extract_photo_evidence,
    load_retained_frames,
)
from ailias.head_reconstruction.evidence.reviewed_photos import prepare_reviewed_photos
from ailias.head_reconstruction.evidence.fit_comparison import write_fit_overlays
from ailias.head_reconstruction.geometry.artifact_io import export_obj, save_fit
from ailias.head_reconstruction.geometry.head_model import HeadModel, model_assets
from ailias.head_reconstruction.geometry.identity_fitting import fit_multiview_identity
from ailias.head_reconstruction.geometry.identity_search import search_identity_fit
from ailias.head_reconstruction.geometry.optimization import prepare_views
from ailias.head_reconstruction.geometry.photo_camera import perspective_camera
from ailias.metahuman.character_features import (
    apply_character_features,
    character_package,
)
from ailias.metahuman.worker_process import execute_unreal_worker
from ailias.generation.unreal import readiness
from ailias.generation.settings import load_settings
from ailias.storage import atomic_json
from ailias.imageprocessing.image_io import read_oriented_image as read_image
from ailias.reanimation.settings import load_policy as load_review_policy
from ailias.reanimation.media import inspect_audio
from ailias.reanimation.pipeline import create_reel, doctor as reel_readiness
from ailias.common.execution import ExecutionChain, PipelineStep, StageJournal
from ailias.resource_management import ResourceManager, finalize_job_resources


class PhotoToCharacterPipeline:
    def __init__(
        self,
        store,
        evidence_path,
        editor,
        project,
        tools,
        policy,
        *,
        video_policy=None,
        appearance_policy=None,
        generation_policy=None,
        photos_run=None,
        search_policy=None,
        review_policy=None,
        assemble=False,
        cloud_services=False,
        audio=None,
        reel_policy=None,
        encoder=None,
    ):
        self.store = store
        self.evidence_path = evidence_path
        self.editor = editor
        self.project = project
        self.tools = tools
        self.fit_policy = policy
        self.video_policy = video_policy
        self.requested_appearance_policy = appearance_policy
        self.generation_policy = generation_policy
        self.photos_run = photos_run
        self.search_policy = search_policy
        self.review_policy = review_policy
        self.assemble = assemble or audio is not None
        self.cloud_services = cloud_services
        self.audio, self.encoder = audio, encoder
        self.reel_policy = reel_policy or (
            load_review_policy() if audio is not None else None
        )
        if audio is not None:
            if encoder is None:
                raise ValueError(
                    "A narrated pipeline requires an explicit FFmpeg executable"
                )
            inspect_audio(audio, self.reel_policy.max_audio_seconds)
            probe = reel_readiness(editor, project, encoder)
            if probe["blockers"]:
                raise ValueError(
                    "Narrated pipeline is blocked: " + "; ".join(probe["blockers"])
                )
        self.root = self.store.resolve(
            str(Path("facefitting") / "fresh-video" / uuid.uuid4().hex)
        )
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = self.root / "manifest.json"
        self.started = time.monotonic()
        self.report = {
            "revision": self.root.name,
            "status": "running",
            "stages": {},
            "artifacts": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "policy": self.fit_policy.record(),
            "scope": "sampled photos to newly created native Character",
            "previous_head_reused": False,
            "previous_fit_reused": False,
            "previous_response_reused": False,
            "previous_appearance_reused": False,
            "blender_used": False,
            "cloud_requested": False,
            "likeness_verified": False,
            "input_dependencies": [],
            "assembly_requested": self.assemble,
            "cloud_services_authorized": cloud_services,
            "narrated_video_requested": audio is not None,
            "sampled_images_reused": True,
            "previous_predictions_reused": False,
            "video_decoding_performed": False,
        }
        if audio is not None:
            self.report["input_dependencies"].append(
                self.store.reference_artifact(audio)
            )
        self.journal = StageJournal(self.report, self.manifest, self.emit_progress)

    def emit_progress(self, event):
        try:
            print(
                json.dumps({"revision": self.root.name, **event}),
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            pass

    def prepare_evidence_and_fit_identity(self):

        def check_unreal_prerequisites():
            probe = readiness(self.editor, self.project)
            if probe["blockers"]:
                raise ValueError(
                    "Unreal prerequisites blocked: " + ", ".join(probe["blockers"])
                )
            self.report["readiness"] = probe
            return probe

        self.journal.execute("native_preflight", check_unreal_prerequisites)
        samples, dependencies = self.journal.execute(
            "load_sampled_photos",
            lambda: load_retained_frames(self.store, self.evidence_path),
        )
        self.report["input_dependencies"].extend(dependencies)
        if self.photos_run is not None:
            photos, photo_dependencies = self.journal.execute(
                "load_reviewed_original_photos",
                lambda: prepare_reviewed_photos(self.store, self.photos_run, self.root),
            )
            samples.extend(photos)
            self.report["input_dependencies"].extend(photo_dependencies)
            self.report["artifacts"]["original_photo_inputs"] = (
                self.store.reference_artifact(
                    self.root / "photo-inputs" / "manifest.json"
                )
            )
        self.model_assets = self.journal.execute(
            "load_models", lambda: model_assets(self.store)
        )
        self.report["model_assets"] = self.model_assets
        self.report["input_dependencies"].extend(
            (a["artifact"] for a in self.model_assets.values())
        )
        self.photo_evidence = self.journal.execute(
            "fresh_photo_processing",
            lambda: extract_photo_evidence(
                self.store,
                samples,
                self.root,
                self.model_assets["landmarker"]["artifact"],
                self.video_policy or VideoPolicy(),
            ),
        )
        self.report["photo_features"] = self.store.reference_artifact(
            self.root / "photo-evidence" / "manifest.json"
        )
        self.report["artifacts"]["photo_features"] = self.report["photo_features"]
        self.report["input_dependencies"].extend(
            (
                self.photo_evidence["parser"][key]
                for key in ("model", "license_artifact")
                if key in self.photo_evidence["parser"]
            )
        )
        self.head_model = HeadModel(
            self.store.artifact_path(self.model_assets["gnm"]["artifact"]),
            self.store.artifact_path(self.model_assets["correspondence"]["artifact"]),
            self.fit_policy.expression_modes,
        )
        self.report["model"] = self.head_model.audit()
        observations = [
            {**frame, "status": "ready"}
            for frame in self.photo_evidence["frames"]
            if "source_xy" in frame
        ]
        entries = {f["face_id"]: {} for f in observations}
        self.fit_views, excluded = prepare_views(self.head_model, observations, entries)
        self.report["inputs"] = {
            "images": len(self.photo_evidence["frames"]),
            "fit_views": len(self.fit_views),
            "excluded": excluded,
            "strict_neutral_views": sum(
                (
                    frame.get("eligible_identity", False)
                    for frame in self.photo_evidence["frames"]
                )
            ),
            "quality_warnings": dict(
                Counter(
                    (
                        reason
                        for frame in self.photo_evidence["frames"]
                        for reason in frame.get("reasons", [])
                    )
                )
            ),
        }
        with redirect_stdout(sys.stderr):
            if self.search_policy is None:
                self.fit_result = self.journal.execute(
                    "fresh_shared_identity_fit",
                    lambda: fit_multiview_identity(
                        self.head_model, self.fit_views, self.fit_policy, self.report
                    ),
                )
            else:
                candidates = self.root / "fit-search"
                candidates.mkdir()

                def retain_candidate(record, candidate):
                    name = f"candidate-{record['index']:03d}"
                    if candidate is not None:
                        path = candidates / (name + ".npz")
                        save_fit(path, candidate)
                        self.report["artifacts"][name] = self.store.reference_artifact(
                            path
                        )
                    atomic_json(candidates / (name + ".json"), record)
                    atomic_json(self.manifest, self.report)
                    self.emit_progress(
                        {
                            "candidate": record["index"],
                            "status": record["status"],
                            "identity_modes": record["identity_modes"],
                            "expression_modes": record["policy"]["expression_modes"],
                            "score": record.get("score"),
                        }
                    )

                self.report["requested_policy"] = self.fit_policy.record()
                self.head_model, self.fit_result, self.fit_policy = (
                    self.journal.execute(
                        "search_shared_identity_fit",
                        lambda: search_identity_fit(
                            lambda modes: HeadModel(
                                self.store.artifact_path(
                                    self.model_assets["gnm"]["artifact"]
                                ),
                                self.store.artifact_path(
                                    self.model_assets["correspondence"]["artifact"]
                                ),
                                modes,
                            ),
                            self.fit_views,
                            self.fit_policy,
                            self.search_policy,
                            self.report,
                            retain_candidate,
                        ),
                    )
                )
                self.report["policy"], self.report["model"] = (
                    self.fit_policy.record(),
                    self.head_model.audit(),
                )

    def prepare_head_mesh_and_tracking(self):
        self.report["fit"] = self.fit_result["report"]
        if self.search_policy is not None:
            self.report["artifacts"].update(
                self.journal.execute(
                    "write_fit_comparisons",
                    lambda: write_fit_overlays(
                        self.store,
                        self.root,
                        self.fit_views,
                        self.fit_result,
                        self.photo_evidence,
                    ),
                )
            )
        self.head_vertices = self.head_model.neutral(self.fit_result["identity"])
        if not np.isfinite(self.head_vertices).all():
            raise ValueError("Fresh identity has nonfinite geometry")
        save_fit(self.root / "fit.npz", self.fit_result)
        export_obj(
            self.root / "neutral-head.obj",
            self.head_vertices,
            self.head_model.triangles,
        )
        with np.load(
            self.store.artifact_path(self.model_assets["gnm"]["artifact"]),
            allow_pickle=False,
        ) as gnm:
            triangle_uvs = gnm["triangle_uvs"].copy()
        self.appearance_policy = (
            self.requested_appearance_policy
            if self.requested_appearance_policy is not None
            else load_appearance_policy()
        )
        generation = (
            self.generation_policy
            if self.generation_policy is not None
            else load_settings()[0]
        )
        self.report["appearance_policy"] = self.appearance_policy
        self.report["blender_manifest"] = str(self.root / "blender" / "manifest.json")
        blender = self.journal.execute(
            "create_blender_appearance",
            lambda: create_blender_head(
                self.store,
                self.root / "blender",
                self.head_model,
                self.fit_result,
                self.fit_views,
                self.photo_evidence,
                triangle_uvs,
                self.tools,
                generation,
                self.appearance_policy,
            ),
        )
        self.report["blender"] = blender
        self.report["blender_used"] = True
        self.report["artifacts"].update(
            {"blender_" + key: value for key, value in blender["artifacts"].items()}
        )
        self.native_mesh = self.store.artifact_path(
            blender["artifacts"]["head_only_mesh"]
        )
        self.expected_native_vertices = self.head_vertices[:, [2, 0, 1]] * [
            100,
            -100,
            100,
        ]
        displacement = np.linalg.norm(
            self.head_vertices - self.head_model.vertices, axis=1
        )
        self.report["geometry"] = {
            "identity_modes": len(self.fit_result["identity"]),
            "vertices": len(self.head_vertices),
            "maximum_template_displacement_cm": float(displacement.max() * 100),
            "median_template_displacement_cm": float(np.median(displacement) * 100),
            "p95_template_displacement_cm": float(
                np.quantile(displacement, 0.95) * 100
            ),
            "vertices_moved_over_1mm_model_units": int((displacement > 0.001).sum()),
            "constraints": "shared identity basis fitted to supported 2D landmark correspondences; not dense measured vertices",
            "physical_scale_calibrated": False,
        }
        by_id = {f["face_id"]: f for f in self.photo_evidence["frames"]}
        fronts = [
            i for i, v in enumerate(self.fit_views) if v["view"]["category"] == "front"
        ]
        if not fronts:
            raise ValueError("Fresh native conform requires a fitted frontal photo")
        index = max(
            fronts,
            key=lambda i: by_id[self.fit_views[i]["face_id"]].get(
                "quality_score", -1000000000.0
            ),
        )
        frame = by_id[self.fit_views[index]["face_id"]]
        tracking, _ = read_image(self.store.artifact_path(frame["image"]))
        with tracking:
            tracking.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            tracking.save(self.root / "tracking-view.png")
            with tracking.convert("RGBA") as rgba:
                (self.root / "tracking-rgba8.bin").write_bytes(rgba.tobytes())
            camera = perspective_camera(
                {
                    "camera": self.fit_result["per_view"][index, :7],
                    "source_size": frame["source_size"],
                },
                self.head_model.center,
                tracking.size,
            )
        atomic_json(self.root / "tracking-camera.json", camera)
        self.report["tracking_face_id"] = frame["face_id"]
        self.character_worker = self.store.copy_artifact(
            module_source("ailias.metahuman.character_creation_worker"),
            str(self.root.relative_to(self.store.root) / "worker"),
        )
        self.conform_worker = self.store.copy_artifact(
            module_source("ailias.metahuman.head_conforming_worker"),
            str(self.root.relative_to(self.store.root) / "worker"),
        )
        self.report["artifacts"].update(
            worker=self.character_worker, conform_worker=self.conform_worker
        )
        self.tools.paths["unreal"] = Path(self.editor)

    def create_and_verify_native_character(self):

        def execute_character_operation(operation, extra):
            return execute_unreal_worker(
                self.tools,
                self.project,
                self.store.artifact_path(self.character_worker),
                self.root,
                operation,
                extra,
                self.report,
            )

        self.created_character = self.journal.execute(
            "fresh_native_conform",
            lambda: execute_character_operation(
                "create",
                {
                    "obj": str(self.native_mesh),
                    "destination": f"/Game/Ailias/FreshVideo/V{self.root.name}",
                    "camera": str(self.root / "tracking-camera.json"),
                    "rgba": str(self.root / "tracking-rgba8.bin"),
                    "conform_worker": str(
                        self.store.artifact_path(self.conform_worker)
                    ),
                    "geometry_output": str(self.root / "native-geometry-before.json"),
                    "expected_bounds": [
                        self.expected_native_vertices.min(0).tolist(),
                        self.expected_native_vertices.max(0).tolist(),
                    ],
                },
            ),
        )
        if self.created_character["status"] != "head_conformed_awaiting_review":
            raise ValueError("Fresh native conform did not save a new Character")
        self.report["character"] = self.created_character["character"]
        self.surface_worker = self.store.copy_artifact(
            module_source("ailias.metahuman.skin_texture_worker"),
            str(self.root.relative_to(self.store.root) / "worker"),
        )
        self.report["artifacts"]["surface_worker"] = self.surface_worker
        alignment = estimate_photo_alignment(self.created_character)
        self.report["photo_projection_alignment"] = alignment
        self.verified_character = self.journal.execute(
            "independent_native_reload",
            lambda: execute_character_operation(
                "verify",
                {
                    "character": self.created_character["character"],
                    "expected": self.created_character["face_landmarks_saved"],
                    "expected_surface": str(self.root / "native-geometry-before.json"),
                    "surface_destination": f"/Game/Ailias/FreshVideo/V{self.root.name}/ReloadedSurface",
                    "surface_worker": str(
                        self.store.artifact_path(self.surface_worker)
                    ),
                    "native_uv_output": str(self.root / "native-skin-uvs.json"),
                    "canonical_to_target": alignment,
                    "conform_worker": str(
                        self.store.artifact_path(self.conform_worker)
                    ),
                },
            ),
        )
        if self.verified_character["status"] != "fresh_character_reloaded":
            raise ValueError("Fresh Character reload failed")
        self.report["artifacts"]["character"] = self.store.reference_artifact(
            character_package(self.project, self.created_character["character"])
        )
        self.preview_worker = self.store.copy_artifact(
            module_source("ailias.metahuman.character_portrait_worker"),
            str(self.root.relative_to(self.store.root) / "worker"),
        )
        self.report["artifacts"].update(
            surface_worker=self.surface_worker, preview_worker=self.preview_worker
        )

    def apply_native_appearance(self):

        def execute_texture_operation(operation, parameters):
            return execute_unreal_worker(
                self.tools,
                self.project,
                self.store.artifact_path(self.surface_worker),
                self.root,
                operation,
                parameters,
                self.report,
            )

        self.report["native_skin_uvs"] = self.verified_character["native_skin_uvs"]
        appearance_sources = [self.report["photo_features"]]
        native_appearance = self.appearance_policy.get("native_appearance")
        if native_appearance is None:
            baked = self.journal.execute(
                "bake_fresh_photo_colour",
                lambda: bake_photo_skin_texture(
                    self.store,
                    self.root,
                    self.root / "native-skin-uvs.json",
                    self.head_model,
                    self.fit_result,
                    self.fit_views,
                    self.photo_evidence,
                    self.appearance_policy["device"],
                    texture_size=self.appearance_policy["texture_size"],
                    maximum_views=self.appearance_policy["max_views"],
                    photo_size=self.appearance_policy["photo_size"],
                ),
            )
            self.report["photo_colour"] = baked
            imported = self.journal.execute(
                "import_native_photo_texture",
                lambda: execute_texture_operation(
                    "import_texture",
                    {
                        "image": baked["texture"],
                        "destination": f"/Game/Ailias/FreshVideo/V{self.root.name}/PhotoTexture",
                    },
                ),
            )
            texture = imported["texture"]
            appearance_sources.extend(
                [
                    self.store.reference_artifact(Path(baked["texture"])),
                    self.store.reference_artifact(
                        character_package(self.project, texture)
                    ),
                ]
            )
        else:
            if not self.appearance_policy.get("native_hair_asset"):
                raise ValueError(
                    "Native appearance without photo baking requires native_hair_asset"
                )
            baked, texture = ({}, None)
            self.report["photo_colour"] = {
                "mode": "native_skin",
                "photo_projection_used": False,
            }

        def map_appearance_evidence():
            return build_appearance_profile(
                self.photo_evidence,
                baked,
                texture,
                appearance_sources,
                native_hair_asset=self.appearance_policy.get("native_hair_asset"),
                assembly=self.appearance_policy.get("assembly"),
                native_appearance=native_appearance,
            )

        profile, mapping = self.journal.execute(
            "derive_native_appearance", map_appearance_evidence
        )
        self.report["appearance_mapping"] = mapping
        profile_path = self.root / "photo-native-profile.json"
        atomic_json(profile_path, profile)

        def apply_appearance_profile():
            result = apply_character_features(
                self.store,
                self.created_character["character"],
                profile_path,
                self.editor,
                self.project,
                self.tools,
                use_working_baseline=False,
                assemble=self.assemble,
                cloud_services=self.cloud_services,
            )
            self.report["native_appearance"] = result
            self.report["cloud_requested"] |= bool(result.get("cloud_requested"))
            if result["status"] != "features_reloaded_awaiting_visual_review":
                raise RuntimeError(
                    "Fresh native appearance did not persist: "
                    + str(result.get("error", result["status"]))
                )
            return result

        self.appearance_result = self.journal.execute(
            "apply_and_reload_native_appearance", apply_appearance_profile
        )
        self.report["character"] = self.appearance_result["character"]
        self.report["artifacts"]["character"] = self.appearance_result["artifacts"][
            "character"
        ]
        if self.assemble:
            blueprint = self.appearance_result.get("blueprint")
            if not blueprint:
                raise RuntimeError(
                    "Native assembly did not return a verified runtime Blueprint"
                )
            self.report["blueprint"] = blueprint
            self.report["artifacts"]["runtime_blueprint"] = (
                self.store.reference_artifact(
                    character_package(self.project, blueprint)
                )
            )

    def render_and_validate_native_preview(self):
        self.report["review_policy"] = (
            self.review_policy
            if self.review_policy is not None
            else load_review_policy(DEFAULT_REVIEW_CONFIG)
        ).to_dict()
        preview = self.journal.execute(
            "render_new_native_preview",
            lambda: execute_unreal_worker(
                self.tools,
                self.project,
                self.store.artifact_path(self.preview_worker),
                self.root,
                "preview",
                {
                    "character": self.appearance_result["character"],
                    "destination": f"/Game/Ailias/FreshVideo/V{self.root.name}/Preview",
                    "output": str(self.root / "native-preview.png"),
                    "warmup_seconds": 20,
                    "lighting": self.report["review_policy"],
                },
                self.report,
            ),
        )
        with Image.open(self.root / "native-preview.png") as image:
            if image.size != (540, 960) or np.asarray(image.convert("RGB")).std() < 2:
                raise RuntimeError(
                    "Native preview is empty or has unexpected dimensions"
                )
        self.report["preview"] = preview
        if not all(
            (self.store.verify(item) for item in self.report["input_dependencies"])
        ):
            raise ValueError("Fresh inputs changed during execution")
        self.report.update(
            status="fresh_video_completed",
            current_stage="awaiting_visual_review",
            fresh_editor_reload_verified=True,
            assembled=bool(self.appearance_result.get("blueprint")),
            limitations=self.report["appearance_mapping"]["limitations"]
            + [
                *(
                    []
                    if self.assemble
                    else ["Editor-native preview is not an assembled animation rig"]
                ),
                "Successful automation and save/reload do not establish personal likeness",
            ],
        )

    def create_narrated_video(self):
        result = create_reel(
            self.store,
            self.audio,
            self.editor,
            self.project,
            self.reel_policy,
            blueprint=self.report["blueprint"],
            encoder=self.encoder,
            tools=self.tools,
        )
        self.report["reanimation"] = result
        if result.get(
            "status"
        ) != "video_rendered_awaiting_visual_review" or not result.get(
            "video_verified"
        ):
            raise RuntimeError(
                "Narrated rendering did not pass its native and media gates"
            )
        if not all(
            self.store.verify(reference)
            for reference in [
                *self.report["input_dependencies"],
                self.report["artifacts"]["runtime_blueprint"],
            ]
        ):
            raise ValueError("Narrated pipeline inputs or assembled Blueprint changed")
        self.report["artifacts"]["video"] = result["artifacts"]["video"]
        manifest = self.store.resolve(
            str(Path("reanimation") / "runs" / result["revision"] / "manifest.json")
        )
        self.report["artifacts"]["reanimation_manifest"] = (
            self.store.reference_artifact(manifest)
        )
        self.report.update(
            status="photo_video_completed_awaiting_visual_review",
            video_verified=True,
            current_stage="awaiting_visual_review",
        )

    def cleanup(self):
        blender_manifest = self.root / "blender" / "manifest.json"
        if blender_manifest.is_file():
            self.report["artifacts"]["blender_manifest"] = (
                self.store.reference_artifact(blender_manifest)
            )
        for path in self.root.iterdir():
            try:
                if (
                    path.is_file()
                    and path != self.manifest
                    and path.name != "tracking-rgba8.bin"
                    and path.stat().st_size
                ):
                    self.report["artifacts"][path.name] = self.store.reference_artifact(
                        path
                    )
            except OSError as error:
                self.report.setdefault("artifact_errors", []).append(
                    {"file": path.name, "type": type(error).__name__}
                )
        if self.report.get("artifact_errors"):
            self.report["status"] = "failed"
        current_stage = self.report.get("current_stage")
        try:
            self.journal.execute(
                "finalize_resources",
                lambda: finalize_job_resources(self.store, self.report, self.manifest),
            )
        finally:
            self.report["current_stage"] = current_stage

    def run(self):
        failure = None
        try:
            with ResourceManager(self.store).defer_retention():
                ExecutionChain(
                    [
                        PipelineStep(
                            "prepare_evidence_and_fit_identity",
                            self.prepare_evidence_and_fit_identity,
                        ),
                        PipelineStep(
                            "prepare_head_mesh_and_tracking",
                            self.prepare_head_mesh_and_tracking,
                        ),
                        PipelineStep(
                            "create_and_verify_native_character",
                            self.create_and_verify_native_character,
                        ),
                        PipelineStep(
                            "apply_native_appearance", self.apply_native_appearance
                        ),
                        PipelineStep(
                            "render_and_validate_native_preview",
                            self.render_and_validate_native_preview,
                        ),
                        *(
                            [
                                PipelineStep(
                                    "create_narrated_video", self.create_narrated_video
                                )
                            ]
                            if self.audio is not None
                            else []
                        ),
                    ]
                ).run(self.journal)
        except BaseException as error:
            failure = error
            self.report.update(
                status="failed" if isinstance(error, Exception) else "interrupted",
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "cause": str(error.__cause__)
                    if error.__cause__ is not None
                    else None,
                    "notes": getattr(error, "__notes__", []),
                },
                project_may_contain_partial_outputs="fresh_native_conform"
                in self.report["stages"],
            )
            if not isinstance(error, Exception):
                raise
        finally:
            try:
                self.cleanup()
            except Exception as error:
                self.report["cleanup_error_type"] = type(error).__name__
                if failure is None:
                    failure = error
                    self.report.update(
                        status="failed",
                        error={"type": type(error).__name__, "message": str(error)},
                    )
                else:
                    failure.add_note(
                        f"Pipeline cleanup also failed: {type(error).__name__}"
                    )
            self.report["seconds"] = round(time.monotonic() - self.started, 6)
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.journal.persist(failure)
        return self.report


def run_photo_to_character_pipeline(
    store,
    evidence_path,
    editor,
    project,
    tools,
    policy,
    *,
    video_policy=None,
    appearance_policy=None,
    generation_policy=None,
    photos_run=None,
    search_policy=None,
    review_policy=None,
    assemble=False,
    cloud_services=False,
    audio=None,
    reel_policy=None,
    encoder=None,
):
    return PhotoToCharacterPipeline(
        store,
        evidence_path,
        editor,
        project,
        tools,
        policy,
        video_policy=video_policy,
        appearance_policy=appearance_policy,
        generation_policy=generation_policy,
        photos_run=photos_run,
        search_policy=search_policy,
        review_policy=review_policy,
        assemble=assemble,
        cloud_services=cloud_services,
        audio=audio,
        reel_policy=reel_policy,
        encoder=encoder,
    ).run()
