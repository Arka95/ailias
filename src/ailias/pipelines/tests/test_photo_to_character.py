"""Pipeline boundary failures remain persistent, specific and non-successful."""

import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.pipelines.photo_to_character import run_photo_to_character_pipeline
from ailias.head_reconstruction.geometry.artifact_io import export_obj
from ailias.metahuman.character_features import character_package
from ailias.storage import Store, atomic_json


class PhotoToCharacterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name), "subject")
        self.store.root.mkdir()
        image = self.store.resolve("retained.png")
        Image.new("RGB", (16, 12)).save(image)
        self.manifest = self.store.resolve("sampled.json")
        self.reference = self.store.reference_artifact(image)
        atomic_json(
            self.manifest,
            {
                "status": "video_evidence_ready",
                "frames": [
                    {
                        "face_id": "frame",
                        "image": self.reference,
                        "source_id": "video",
                    }
                ],
            },
        )
        self.policy = SimpleNamespace(record=lambda: {})

    def invoke(self):
        return run_photo_to_character_pipeline(
            self.store,
            self.manifest,
            Path("Editor.exe"),
            Path("Project.uproject"),
            SimpleNamespace(),
            self.policy,
        )

    def persisted(self, result):
        return json.loads(
            self.store.resolve(
                "facefitting\\fresh-video\\" + result["revision"] + "\\manifest.json"
            ).read_text()
        )

    def test_blocked_preflight_is_a_failed_stage_and_never_starts_inference_or_decoding(
        self,
    ):
        with (
            patch(
                "ailias.pipelines.photo_to_character.readiness",
                return_value={"blockers": ["missing_editor"]},
            ),
            patch(
                "ailias.pipelines.photo_to_character.extract_photo_evidence"
            ) as process,
        ):
            result = self.invoke()
        self.assertEqual(self.persisted(result)["status"], "failed")
        self.assertEqual(result["stages"]["native_preflight"]["status"], "failed")
        self.assertIn("missing_editor", result["error"]["message"])
        self.assertFalse(result["video_decoding_performed"])
        process.assert_not_called()

    def test_photo_stage_exception_preserves_the_root_cause_and_stops_before_fitting(
        self,
    ):
        try:
            raise OSError("model read failed")
        except OSError as cause:
            error = RuntimeError("photo failed")
            error.__cause__ = cause
        with (
            patch(
                "ailias.pipelines.photo_to_character.readiness",
                return_value={"blockers": []},
            ),
            patch(
                "ailias.pipelines.photo_to_character.model_assets",
                return_value={"landmarker": {"artifact": self.reference}},
            ),
            patch(
                "ailias.pipelines.photo_to_character.extract_photo_evidence",
                side_effect=error,
            ),
            patch("ailias.pipelines.photo_to_character.fit_multiview_identity") as fit,
        ):
            result = self.invoke()
        saved = self.persisted(result)
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["current_stage"], "fresh_photo_processing")
        self.assertEqual(saved["error"]["cause"], "model read failed")
        self.assertEqual(saved["stages"]["fresh_photo_processing"]["status"], "failed")
        self.assertTrue(self.store.verify(self.reference))
        fit.assert_not_called()

    def complete(
        self,
        *,
        appearance_fails=False,
        blender_fails=False,
        review_policy=None,
        native_hair_asset=None,
        assembly=None,
        native_appearance=None,
        narrated=False,
        assembly_fails=False,
        reel_fails=False,
    ):
        project = self.store.resolve("Project\\Preview.uproject")
        project.parent.mkdir()
        project.write_text("{}")
        archive = self.store.resolve("generic-model.npz")
        np.savez(archive, triangle_uvs=np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]))
        assets = {
            name: {"artifact": self.store.reference_artifact(archive)}
            for name in ("gnm", "correspondence", "landmarker")
        }
        colour = {"status": "observed", "median_linear_rgb": [0.1, 0.05, 0.025]}
        region = {
            "status": "estimated",
            "dark_fraction": 0.4,
            "presence_estimate": True,
            "median_dark_linear_rgb": [0.02, 0.015, 0.01],
        }
        frame = {
            "face_id": "frame",
            "image": self.reference,
            "source_size": [16, 12],
            "source_xy": [[1.0, 1.0]],
            "quality_score": 1,
        }
        evidence = {
            "frames": [frame],
            "parser": {},
            "faces": {
                "frame": {
                    "status": "observed",
                    "features": {
                        "iris": colour,
                        "regions": {"scalp_hair": colour, "eyebrows": colour},
                        "facial_hair_appearance": {
                            "regions": {"beard": region, "moustache": region}
                        },
                    },
                }
            },
        }
        vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.1]])
        model = SimpleNamespace(
            vertices=vertices,
            neutral=lambda identity: vertices,
            triangles=np.array([[0, 1, 2]]),
            center=np.zeros(3),
            audit=lambda: {"correspondences": 468},
        )
        fitted = {
            "identity": np.zeros(1),
            "per_view": np.zeros((1, 8)),
            "report": {"views": [{"face_id": "frame", "supported_points": 120}]},
        }
        source = "/Game/Fresh/Character.Character"
        candidate = "/Game/CurrentAppearance/Character.Character"

        def asset(path):
            package = character_package(project, path)
            package.parent.mkdir(parents=True, exist_ok=True)
            package.write_bytes(b"new native asset")
            return self.store.reference_artifact(package)

        operations = []

        def blender(store, root, *args):
            operations.append("blender")
            if blender_fails:
                raise RuntimeError("Blender export failed")
            root.mkdir()
            mesh = root / "head-for-unreal.obj"
            export_obj(mesh, vertices[:, [2, 0, 1]] * 100, model.triangles)
            return {
                "status": "blender_head_created",
                "artifacts": {"head_only_mesh": store.reference_artifact(mesh)},
            }

        def process(store, samples, root, *args):
            output = root / "photo-evidence"
            output.mkdir()
            atomic_json(output / "manifest.json", evidence)
            return evidence

        def native(tools, project, worker, root, operation, parameters, report):
            operations.append(operation)
            if operation == "create":
                self.assertEqual(Path(parameters["obj"]).parent.name, "blender")
                self.assertTrue(Path(parameters["obj"]).is_file())
                asset(source)
                return {
                    "status": "head_conformed_awaiting_review",
                    "character": source,
                    "face_landmarks_saved": vertices.tolist(),
                    "face_landmarks_solved_in_target_pose": vertices.tolist(),
                    "face_landmarks_after": (vertices + [0, 0, 100]).tolist(),
                }
            if operation == "verify":
                return {"status": "fresh_character_reloaded", "native_skin_uvs": {}}
            if operation == "import_texture":
                asset("/Game/Current/PhotoBaseColor")
                return {
                    "status": "native_photo_texture_imported",
                    "texture": "/Game/Current/PhotoBaseColor",
                }
            self.preview_parameters = parameters
            image = Image.new("RGB", (540, 960), (40, 40, 40))
            image.paste((180, 130, 100), (100, 100, 400, 800))
            image.save(parameters["output"])
            return {"status": "native_editor_preview_rendered"}

        self.baked_photo_texture = False

        def bake(store, root, *args, **kwargs):
            self.baked_photo_texture = True
            image = root / "native-photo-basecolor.png"
            Image.new("RGB", (32, 32), (100, 80, 60)).save(image)
            return {"texture": str(image), "hair_length_model_units": 0.02}

        def apply(*args, **kwargs):
            self.assertFalse(kwargs["use_working_baseline"])
            self.native_profile = json.loads(args[2].read_text())
            result = (
                {"status": "failed", "error": "groom missing"}
                if appearance_fails
                else {
                    "status": "features_reloaded_awaiting_visual_review",
                    "character": candidate,
                    "artifacts": {"character": asset(candidate)},
                }
            )
            if kwargs["assemble"]:
                operations.append("assemble")
                if assembly_fails:
                    return {"status": "failed", "error": "assembly failed"}
                result["blueprint"] = "/Game/CurrentAppearance/Runtime/BP.BP"
                result["artifacts"]["blueprint"] = asset(result["blueprint"])
            self.cloud_services_requested = kwargs["cloud_services"]
            return result

        def reanimate(store, audio, editor, project, policy, **kwargs):
            operations.append("reanimation")
            self.assertEqual(
                kwargs["blueprint"], "/Game/CurrentAppearance/Runtime/BP.BP"
            )
            self.assertEqual(policy.exposure_compensation_ev, -5.5)
            root = store.resolve("reanimation\\runs\\" + "f" * 32)
            root.mkdir(parents=True)
            video = root / "portrait-reel.mp4"
            video.write_bytes(b"verified synthetic video")
            result = {
                "revision": root.name,
                "status": "failed"
                if reel_fails
                else "video_rendered_awaiting_visual_review",
                "video_verified": not reel_fails,
                "artifacts": {"video": store.reference_artifact(video)},
            }
            atomic_json(root / "manifest.json", result)
            return result

        voice = None
        if narrated:
            voice = self.store.resolve("narration.wav")
            with wave.open(str(voice), "wb") as stream:
                stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x01\x00" * 2400)

        with ExitStack() as stack:
            for name, value in {
                "readiness": lambda *args: {"blockers": []},
                "model_assets": lambda *args: assets,
                "extract_photo_evidence": process,
                "HeadModel": lambda *args: model,
                "prepare_views": lambda *args: (
                    [{"face_id": "frame", "view": {"category": "front"}}],
                    [],
                ),
                "fit_multiview_identity": lambda *args: fitted,
                "save_fit": lambda *args: None,
                "create_blender_head": blender,
                "execute_unreal_worker": native,
                "bake_photo_skin_texture": bake,
                "apply_character_features": apply,
                "reel_readiness": lambda *args: {"blockers": []},
                "create_reel": reanimate,
            }.items():
                stack.enter_context(
                    patch(
                        "ailias.pipelines.photo_to_character." + name, side_effect=value
                    )
                )
            result = run_photo_to_character_pipeline(
                self.store,
                self.manifest,
                Path("Editor.exe"),
                project,
                SimpleNamespace(paths={}),
                SimpleNamespace(record=lambda: {}, expression_modes=[]),
                generation_policy=SimpleNamespace(),
                review_policy=review_policy,
                appearance_policy={
                    "device": "cuda:0",
                    "texture_size": 256,
                    "max_views": 1,
                    "photo_size": 256,
                    "hair_strands": 0,
                    "native_hair_asset": native_hair_asset,
                    "assembly": assembly,
                    "native_appearance": native_appearance,
                },
                audio=voice,
                encoder=Path("ffmpeg.exe") if narrated else None,
            )
        return result, operations

    def test_one_invocation_reaches_fresh_texture_grooms_and_a_new_preview_without_old_baseline(
        self,
    ):
        result, operations = self.complete()
        self.assertEqual(self.persisted(result)["status"], "fresh_video_completed")
        self.assertEqual(
            operations, ["blender", "create", "verify", "import_texture", "preview"]
        )
        self.assertTrue(result["blender_used"])
        self.assertFalse(result["assembled"])
        self.assertFalse(result["likeness_verified"])
        self.assertEqual(result["fit"]["views"][0]["supported_points"], 120)
        self.assertEqual(result["model"]["correspondences"], 468)
        self.assertEqual(result["geometry"]["vertices_moved_over_1mm_model_units"], 0)
        self.assertTrue(self.store.verify(self.reference))

    def test_native_appearance_failure_cannot_skip_to_preview_or_success(self):
        result, operations = self.complete(appearance_fails=True)
        self.assertEqual(self.persisted(result)["status"], "failed")
        self.assertEqual(result["current_stage"], "apply_and_reload_native_appearance")
        self.assertIn("groom missing", result["error"]["message"])
        self.assertNotIn("preview", operations)

    def test_native_complexion_skips_photo_atlas_baking_and_import(self):
        settings = {"skin": {"u": 0.35, "v": 0.9, "face_texture_index": 7}}
        result, operations = self.complete(
            native_appearance=settings,
            native_hair_asset="/MetaHumanCharacter/Grooms/SelectedHair",
        )
        self.assertEqual(result["status"], "fresh_video_completed")
        self.assertFalse(self.baked_photo_texture)
        self.assertNotIn("import_texture", operations)
        self.assertEqual(self.native_profile["skin"], settings["skin"])
        self.assertIsNone(self.native_profile["face_basecolor"])
        self.assertFalse(result["photo_colour"]["photo_projection_used"])

    def test_review_lighting_and_native_hair_are_independent_of_reanimation_and_blender(
        self,
    ):
        from ailias.reanimation.settings import ReelPolicy, load_policy

        original_reanimation = load_policy().to_dict()
        lighting = ReelPolicy(exposure_compensation_ev=-3)
        hair = "/MetaHumanCharacter/Optional/Grooms/Bindings/Hair/WI_Hair_S_SweptUp"
        result, _ = self.complete(
            review_policy=lighting,
            native_hair_asset=hair,
            assembly={"pipeline": "cinematic"},
        )
        self.assertEqual(result["status"], "fresh_video_completed")
        self.assertEqual(
            self.preview_parameters["lighting"]["exposure_compensation_ev"], -3
        )
        self.assertEqual(self.native_profile["grooms"]["Hair"]["asset"], hair)
        self.assertEqual(self.native_profile["assembly"], {"pipeline": "cinematic"})
        self.assertEqual(result["appearance_policy"]["hair_strands"], 0)
        self.assertEqual(load_policy().to_dict(), original_reanimation)

    def test_closed_progress_stream_does_not_abort_a_successful_pipeline(self):
        with patch("ailias.pipelines.photo_to_character.sys.stderr") as stream:
            stream.write.side_effect = OSError(22, "closed progress stream")
            result, _ = self.complete()
        self.assertEqual(result["status"], "fresh_video_completed")

    def test_blender_failure_cannot_be_skipped_or_launch_unreal_creation(self):
        result, operations = self.complete(blender_fails=True)
        self.assertEqual(self.persisted(result)["status"], "failed")
        self.assertEqual(result["current_stage"], "create_blender_appearance")
        self.assertEqual(operations, ["blender"])
        self.assertIn("Blender export failed", result["error"]["message"])

    def test_narrated_chain_assembles_the_new_character_and_publishes_verified_video(
        self,
    ):
        result, operations = self.complete(narrated=True)
        saved = self.persisted(result)
        self.assertEqual(
            saved["status"], "photo_video_completed_awaiting_visual_review"
        )
        self.assertEqual(operations[-3:], ["assemble", "preview", "reanimation"])
        self.assertTrue(saved["assembled"] and saved["video_verified"])
        self.assertTrue(self.store.verify(saved["artifacts"]["video"]))
        self.assertTrue(self.store.verify(saved["artifacts"]["reanimation_manifest"]))
        self.assertFalse(self.cloud_services_requested)
        self.assertFalse(saved["likeness_verified"])

    def test_failed_assembly_never_reaches_rendering(self):
        result, operations = self.complete(narrated=True, assembly_fails=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_stage"], "apply_and_reload_native_appearance")
        self.assertNotIn("reanimation", operations)

    def test_failed_reel_never_claims_verified_video(self):
        result, _ = self.complete(narrated=True, reel_fails=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_stage"], "create_narrated_video")
        self.assertFalse(result.get("video_verified", False))

    def test_total_duration_includes_finalization(self):
        now = [100.0]

        def finalize(*args):
            now[0] += 7.0

        with (
            patch(
                "ailias.pipelines.photo_to_character.time.monotonic",
                side_effect=lambda: now[0],
            ),
            patch(
                "ailias.pipelines.photo_to_character.finalize_job_resources",
                side_effect=finalize,
            ),
        ):
            result, _ = self.complete()
        self.assertEqual(result["seconds"], 7.0)
        self.assertEqual(result["stages"]["finalize_resources"]["seconds"], 7.0)

    def test_narrated_cli_forwards_voice_and_explicit_cloud_permission_with_independent_lighting(
        self,
    ):
        from ailias.cli import parser
        from ailias.facefitting.cli import execute

        args = parser().parse_args(
            [
                "facefitting",
                "video-create",
                "subject",
                "--evidence",
                str(self.manifest),
                "--editor",
                "Editor.exe",
                "--project",
                "Project.uproject",
                "--voice",
                "narration.wav",
                "--ffmpeg",
                "ffmpeg.exe",
                "--cloud-services",
            ]
        )
        with (
            patch(
                "ailias.pipelines.photo_to_character.run_photo_to_character_pipeline",
                return_value={"status": "photo_video_completed_awaiting_visual_review"},
            ) as pipeline,
            patch("ailias.generation.tools.NativeTools"),
        ):
            _, code = execute(args, self.store.root.parent)
        self.assertEqual(code, 0)
        self.assertEqual(pipeline.call_args.kwargs["audio"], Path("narration.wav"))
        self.assertTrue(pipeline.call_args.kwargs["cloud_services"])
        self.assertNotEqual(
            pipeline.call_args.kwargs["review_policy"].exposure_compensation_ev,
            pipeline.call_args.kwargs["reel_policy"].exposure_compensation_ev,
        )

    def test_missing_narration_encoder_fails_before_creating_a_fitting_job(self):
        with self.assertRaisesRegex(ValueError, "FFmpeg"):
            run_photo_to_character_pipeline(
                self.store,
                self.manifest,
                Path("Editor.exe"),
                Path("Project.uproject"),
                SimpleNamespace(),
                self.policy,
                audio=Path("narration.wav"),
            )
        self.assertFalse(self.store.resolve("facefitting\\fresh-video").exists())
