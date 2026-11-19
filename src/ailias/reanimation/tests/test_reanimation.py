"""Test portrait math, exact media gates and persisted native outcomes without inference or rendering."""

import array
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from ailias.reanimation.media import (
    encode_command,
    inspect_audio,
    inspect_frames,
    verify_video,
)
from ailias.reanimation.pipeline import REQUIRED_PLUGINS, create_reel, inspect_job
from ailias.reanimation.settings import ReelPolicy
from ailias.storage import Store, atomic_json
from ailias.common.execution import StageJournal


class ReelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def audio(self, silent=False):
        path = self.root / "voice.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            stream.writeframes(
                array.array("h", [0 if silent else 1000] * 2400).tobytes()
            )
        return path

    def cli_args(self, extra=(), alias="subject", audio_flag="--audio"):
        from ailias.cli import parser

        return parser().parse_args(
            [
                "reanimation",
                "run",
                *([alias] if alias else []),
                audio_flag,
                str(self.root / "voice.wav"),
                "--editor",
                str(self.root / "UnrealEditor-Cmd.exe"),
                "--project",
                str(self.root / "Review.uproject"),
                *extra,
            ]
        )

    def test_named_and_legacy_aliases_select_the_same_store(self):
        from ailias.reanimation.cli import execute

        with patch(
            "ailias.reanimation.pipeline.create_reel",
            return_value={"status": "audio_solved_awaiting_rigged_metahuman"},
        ) as native:
            for alias, extra in (
                ("subject", []),
                (None, ["--alias", "subject"]),
                ("subject", ["--alias", "subject"]),
            ):
                execute(self.cli_args(extra, alias), self.root / "data")
                self.assertEqual(
                    native.call_args.args[0].root,
                    self.root.resolve() / "data" / "subject",
                )

    def test_voice_and_audio_flags_supply_the_same_external_audio_without_synthesis(
        self,
    ):
        from ailias.reanimation.cli import execute

        audio = self.audio()
        for flag in ("--voice", "--audio"):
            with (
                self.subTest(flag=flag),
                patch.dict("sys.modules", {"mcp": None, "yapit": None}),
                patch(
                    "ailias.reanimation.pipeline.create_reel",
                    return_value={"status": "audio_solved_awaiting_rigged_metahuman"},
                ) as native,
            ):
                args = self.cli_args(
                    ["--alias", "subject", "--resolution", "540x960"],
                    alias=None,
                    audio_flag=flag,
                )
                self.assertEqual(execute(args, self.root / "data")[1], 0)
                self.assertEqual(native.call_args.args[1], audio)
                self.assertEqual(native.call_args.args[4].width, 540)
                self.assertEqual(
                    native.call_args.kwargs, {"blueprint": None, "encoder": None}
                )

    def test_missing_or_conflicting_alias_never_starts_native_work(self):
        from ailias.reanimation.cli import execute

        with patch("ailias.reanimation.pipeline.create_reel") as native:
            for alias, extra in ((None, []), ("subject", ["--alias", "different"])):
                with self.subTest(alias=alias), self.assertRaises(ValueError):
                    execute(self.cli_args(extra, alias), self.root / "data")
            native.assert_not_called()
        self.assertFalse((self.root / "data").exists())

    def test_named_alias_works_with_status_job_positional(self):
        from ailias.cli import parser
        from ailias.reanimation.cli import execute

        with patch(
            "ailias.reanimation.pipeline.inspect_job",
            return_value={"integrity_problems": []},
        ) as inspect:
            for arguments in (["subject", "a" * 32], ["--alias", "subject", "a" * 32]):
                args = parser().parse_args(["reanimation", "status", *arguments])
                self.assertEqual(execute(args, self.root / "data")[1], 0)
                self.assertEqual(
                    inspect.call_args.args[0].root,
                    self.root.resolve() / "data" / "subject",
                )
                self.assertEqual(inspect.call_args.args[1], "a" * 32)

    def test_cli_resolution_override_preserves_lighting_and_camera_without_changing_preset(
        self,
    ):
        from ailias.reanimation.cli import execute

        config = self.root / "custom.json"
        atomic_json(config, {"width": 720, "height": 1280, "key_lumens": 750})
        framing = ReelPolicy().framing()
        for extra, dimensions, lumens in (
            ([], (1080, 1920), 900),
            (["--resolution", "540x960"], (540, 960), 900),
            (["--reel-config", str(config)], (720, 1280), 750),
            (
                ["--reel-config", str(config), "--resolution", "540X960"],
                (540, 960),
                750,
            ),
        ):
            with (
                self.subTest(extra=extra),
                patch(
                    "ailias.reanimation.pipeline.create_reel",
                    return_value={"status": "audio_solved_awaiting_rigged_metahuman"},
                ) as native,
            ):
                _, code = execute(self.cli_args(extra), self.root / "data")
                policy = native.call_args.args[4]
                self.assertEqual(code, 0)
                self.assertEqual((policy.width, policy.height), dimensions)
                self.assertEqual(policy.key_lumens, lumens)
                self.assertEqual(policy.framing(), framing)
        self.assertEqual(json.loads(config.read_text())["width"], 720)

    def test_invalid_cli_resolution_never_launches_native_or_creates_a_job(self):
        from ailias.reanimation.cli import execute

        with patch("ailias.reanimation.pipeline.create_reel") as native:
            for value in ("bad", "1920x1080", "541x960", "1080x0", "1080x1920x30"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    execute(self.cli_args(["--resolution", value]), self.root / "data")
            native.assert_not_called()
        self.assertFalse((self.root / "data").exists())

    def test_cli_selected_model_audio_and_placement_reach_the_reel_job(self):
        from ailias.reanimation.cli import execute

        model = "/Game/Characters/AnotherPerson/BP_Person"
        audio = self.root / "another-voice.wav"
        args = self.cli_args(
            [
                "--blueprint",
                model,
                "--audio",
                str(audio),
                "--character-position",
                "40",
                "60",
                "0",
                "--character-yaw",
                "30",
                "--screen-position",
                ".45",
                ".52",
            ]
        )
        with patch(
            "ailias.reanimation.pipeline.create_reel",
            return_value={"status": "video_rendered_awaiting_visual_review"},
        ) as native:
            execute(args, self.root / "data")
        self.assertEqual(native.call_args.args[1], audio)
        self.assertEqual(native.call_args.kwargs["blueprint"], model)
        policy = native.call_args.args[4]
        self.assertEqual(policy.character_location_cm, [40, 60, 0])
        self.assertEqual(policy.character_yaw_degrees, 30)
        self.assertEqual(policy.subject_screen_position, [0.45, 0.52])

    def test_camera_shift_maps_normalized_screen_placement_in_rotated_rig_axes(self):
        policy = ReelPolicy(
            character_yaw_degrees=90, subject_screen_position=(0.25, 0.75)
        )
        placement = policy.placement()
        for actual, expected in zip(placement["front"], (-1, 0, 0)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            placement["camera_target_shift_cm"], (0, 7.3125, 13)
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(ReelPolicy().placement()["camera_target_shift_cm"], (0, 0, 0))

    def test_portrait_lens_geometry_preserves_face_field_of_view(self):
        policy = ReelPolicy()
        framing = policy.framing()
        height = (
            2
            * framing["camera_distance_cm"]
            * math.tan(math.radians(framing["vertical_fov_degrees"]) / 2)
        )
        self.assertAlmostEqual(height, 52)
        self.assertAlmostEqual(framing["visible_width_cm"], 29.25)
        self.assertAlmostEqual(framing["camera_distance_cm"], 122.7777778)
        for values in (
            {"width": 1920},
            {"fps": True},
            {"mood_intensity": float("nan")},
            {"background_srgb": [1, 2, 3]},
            {"encoder": "auto"},
            {"exposure_compensation_ev": -13},
            {"exposure_compensation_ev": True},
            {"light_size_cm": 0},
            {"light_size_cm": float("inf")},
            {"character_location_cm": [0, 0]},
            {"character_location_cm": [0, True, 0]},
            {"subject_screen_position": [1.1, 0.5]},
            {"character_yaw_degrees": 181},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ReelPolicy(**values)

    def test_narration_rejects_silence_and_excess_duration(self):
        self.assertAlmostEqual(inspect_audio(self.audio())["duration_seconds"], 0.1)
        with self.assertRaisesRegex(ValueError, "Silent"):
            inspect_audio(self.audio(True))
        with self.assertRaisesRegex(ValueError, "duration"):
            inspect_audio(self.audio(), 0.05)

    def test_frames_require_contiguous_complete_portrait_dimensions(self):
        policy = ReelPolicy(width=90, height=160)
        for index in (0, 1):
            Image.new("RGB", (90, 160)).save(self.root / f"frame.{index:06d}.png")
        self.assertEqual(len(inspect_frames(self.root, policy, 2)), 2)
        (self.root / "frame.000000.png").rename(self.root / "frame.000002.png")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            inspect_frames(self.root, policy, 2)

    def test_gpu_encode_is_explicit_and_never_uses_shortest_or_overwrites(self):
        command = encode_command(
            Path("ffmpeg.exe"),
            self.root,
            self.audio(),
            self.root / "reel.mp4",
            ReelPolicy(),
            3,
        )
        self.assertIn("h264_nvenc", command)
        self.assertIn("-n", command)
        self.assertNotIn("-shortest", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "3")

    def job(self, outcome, blueprint=None, keep_frames=False, encoder=None):
        project = self.root / "Review.uproject"
        atomic_json(
            project,
            {
                "Plugins": [
                    {"Name": name, "Enabled": True}
                    for name in (*REQUIRED_PLUGINS, "AppleProResMedia")
                ]
            },
        )
        editor = self.root / "UnrealEditor-Cmd.exe"
        editor.write_bytes(b"fake")
        store = Store(self.root / "data", "subject")

        def launch(name, operation, args, directory, environment):
            request = json.loads(
                Path(environment["AILIAS_REANIMATION_REQUEST"]).read_text()
            )
            if outcome == "partial_frames":
                for number in range(3):
                    Image.new("RGB", (90, 160)).save(
                        Path(request["frames"]) / f"frame.{number:06d}.png"
                    )
                (directory / "portrait-reel.mp4").write_bytes(b"partial encoding")
                raise ValueError("native render interrupted")
            performance = request["destination"] + "/Performance.Performance"
            if outcome != "missing":
                package = (
                    project.parent
                    / "Content"
                    / Path(*request["destination"].split("/")[2:])
                )
                package.mkdir(parents=True)
                (package / "Performance.uasset").write_bytes(b"fake native package")
                (package / "Narration.uasset").write_bytes(b"fake imported sound")
            result = {
                "status": "audio_solved_awaiting_rigged_metahuman",
                "performance": performance,
                "sound": request["destination"] + "/Narration.Narration",
                "processed_frames": 0 if outcome == "empty" else 3,
            }
            if outcome in {"rendered", "missing_movie"}:
                result.update(
                    status="movie_rendered"
                    if request.get("movie")
                    else "frames_rendered",
                    render_success=True,
                )
                for name, suffix in (
                    ("sequence", ".uasset"),
                    ("map", ".umap"),
                    ("material", ".uasset"),
                ):
                    (package / (name + suffix)).write_bytes(b"native output")
                    result[name] = request["destination"] + "/" + name + "." + name
                if request.get("movie"):
                    if outcome != "missing_movie":
                        Path(request["movie"]).write_bytes(b"synthetic ProRes movie")
                else:
                    for number in range(request["frame_count"]):
                        Image.new(
                            "RGB",
                            (request["policy"]["width"], request["policy"]["height"]),
                        ).save(Path(request["frames"]) / f"frame.{number:06d}.png")
            atomic_json(Path(request["receipt"]), result)
            if outcome == "crash":
                raise ValueError("native exit_3221225477")
            return {"seconds": 0}

        result = create_reel(
            store,
            self.audio(),
            editor,
            project,
            ReelPolicy(),
            blueprint=blueprint,
            tools=SimpleNamespace(paths={}, run=launch),
            keep_frames=keep_frames,
            encoder=encoder,
        )
        return store, result

    def test_failed_render_removes_partial_video_and_interior_frames_but_keeps_diagnostics(
        self,
    ):
        store, result = self.job("partial_frames")
        root = store.artifact_path(result["artifacts"]["first_frame"]).parent
        self.assertEqual(result["status"], "failed")
        self.assertFalse((root.parent / "portrait-reel.mp4").exists())
        self.assertEqual(
            [path.name for path in sorted(root.glob("*.png"))],
            ["frame.000000.png", "frame.000002.png"],
        )
        self.assertEqual(result["cleanup"]["errors"], 0)
        self.assertEqual(
            inspect_job(store, result["revision"])["integrity_problems"], []
        )

    def test_keep_frames_preserves_failed_render_source_frames_when_requested(self):
        store, result = self.job("partial_frames", keep_frames=True)
        root = store.artifact_path(result["artifacts"]["first_frame"]).parent
        self.assertEqual(len(list(root.glob("*.png"))), 3)
        self.assertTrue(result["keep_frames"])

    def test_cleanup_failure_does_not_replace_the_original_native_failure(self):
        with patch(
            "ailias.resource_management.job_resources.cleanup_owned_files",
            side_effect=PermissionError,
        ):
            _, result = self.job("crash")
        self.assertEqual(result["status"], "failed")
        self.assertIn("exit_3221225477", result["error"]["message"])
        self.assertEqual(result["cleanup"]["errors"], 1)

    def test_audio_solve_without_rig_never_claims_video(self):
        store, result = self.job("saved")
        self.assertEqual(result["status"], "audio_solved_awaiting_rigged_metahuman")
        self.assertFalse(result["video_verified"])
        self.assertTrue(store.verify(result["artifacts"]["performance"]))
        self.assertEqual(
            inspect_job(store, result["revision"])["integrity_problems"], []
        )

    def test_success_receipt_without_native_performance_package_is_failure(self):
        _, result = self.job("missing")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["video_verified"])

    def test_empty_animation_is_not_a_successful_audio_solve(self):
        _, result = self.job("empty")
        self.assertEqual(result["status"], "failed")
        self.assertIn("processed animation frames", result["error"]["message"])

    def test_native_receipt_cannot_override_a_process_failure(self):
        _, result = self.job("crash")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["video_verified"])
        self.assertIn("exit_3221225477", result["error"]["message"])
        self.assertEqual(result["native"]["processed_frames"], 3)

    def test_missing_encoder_blocks_video_before_native_execution(self):
        _, result = self.job("saved", "/Game/MetaHumans/Arka/BP_Arka")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("--ffmpeg", " ".join(result["readiness"]["blockers"]))

    def test_video_publication_checks_decoded_frames_and_audio_coverage(self):
        metadata = SimpleNamespace(stderr="Video: h264, 1080x1920, 30 fps\nAudio: aac")
        for count, video_time, audio_time, valid in (
            (3, 100000, 106667, True),
            (2, 100000, 106667, False),
            (3, 500000, 106667, False),
            (3, 100000, 10000, False),
        ):
            video = SimpleNamespace(
                returncode=0, stdout=f"frame={count}\nout_time_us={video_time}\n"
            )
            audio = SimpleNamespace(returncode=0, stdout=f"out_time_us={audio_time}\n")
            with self.subTest(
                count=count, video_time=video_time, audio_time=audio_time
            ):
                with patch(
                    "ailias.reanimation.media.subprocess.run",
                    side_effect=[metadata, video, audio],
                ):
                    if valid:
                        self.assertTrue(
                            verify_video(
                                Path("ffmpeg.exe"), Path("reel.mp4"), ReelPolicy(), 3
                            )["audio_decoded"]
                        )
                    else:
                        with self.assertRaises(ValueError):
                            verify_video(
                                Path("ffmpeg.exe"), Path("reel.mp4"), ReelPolicy(), 3
                            )

    def test_render_completion_yields_shutdown_to_startup_script_owner(self):
        from ailias.reanimation import unreal_worker

        keepalive = []
        unreal = SimpleNamespace(
            EditorPythonScripting=SimpleNamespace(
                set_keep_python_script_alive=keepalive.append
            )
        )
        receipt = self.root / "receipt.json"
        with (
            patch.dict("sys.modules", {"unreal": unreal}),
            patch.object(unreal_worker, "REQUEST", {"receipt": str(receipt)}),
            patch.object(unreal_worker, "RESULT", {}),
        ):
            unreal_worker.finished(None, True)
        self.assertEqual(keepalive, [False])
        self.assertEqual(json.loads(receipt.read_text())["status"], "frames_rendered")

    def test_persistent_render_completion_removes_its_delegate_without_closing_the_editor(
        self,
    ):
        from ailias.reanimation import unreal_worker

        keepalive, removed = [], []
        unreal = SimpleNamespace(
            EditorPythonScripting=SimpleNamespace(
                set_keep_python_script_alive=keepalive.append
            )
        )
        executor = SimpleNamespace(
            on_executor_finished_delegate=SimpleNamespace(
                remove_callable=removed.append
            )
        )
        receipt = self.root / "persistent-receipt.json"
        with (
            patch.dict("sys.modules", {"unreal": unreal}),
            patch.object(unreal_worker, "REQUEST", {"receipt": str(receipt)}),
            patch.object(unreal_worker, "RESULT", {}),
            patch.object(unreal_worker, "KEEP_EDITOR_ALIVE", True),
        ):
            unreal_worker.finished(executor, True)
        self.assertEqual(keepalive, [])
        self.assertEqual(removed, [unreal_worker.finished])
        self.assertEqual(json.loads(receipt.read_text())["status"], "frames_rendered")

    def test_render_delegate_cleanup_failure_cannot_publish_persistent_success(self):
        from ailias.reanimation import unreal_worker
        from unittest.mock import Mock

        unreal = SimpleNamespace(
            EditorPythonScripting=SimpleNamespace(set_keep_python_script_alive=Mock())
        )
        executor = SimpleNamespace(
            on_executor_finished_delegate=SimpleNamespace(
                remove_callable=Mock(side_effect=RuntimeError)
            )
        )
        receipt = self.root / "cleanup-receipt.json"
        with (
            patch.dict("sys.modules", {"unreal": unreal}),
            patch.object(unreal_worker, "REQUEST", {"receipt": str(receipt)}),
            patch.object(unreal_worker, "RESULT", {}),
            patch.object(unreal_worker, "KEEP_EDITOR_ALIVE", True),
        ):
            unreal_worker.finished(executor, True)
        result = json.loads(receipt.read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["render_success"])
        unreal.EditorPythonScripting.set_keep_python_script_alive.assert_not_called()

    def test_host_records_encoding_verification_and_cleanup_as_separate_stages(self):
        encoder = self.root / "ffmpeg.exe"
        encoder.write_bytes(b"synthetic executable")

        def encode(command, **kwargs):
            Path(command[-1]).write_bytes(b"synthetic video")
            return SimpleNamespace(returncode=0)

        with (
            patch("ailias.reanimation.pipeline.subprocess.run", side_effect=encode),
            patch(
                "ailias.reanimation.pipeline.verify_video",
                return_value={"audio_decoded": True},
            ),
        ):
            _, result = self.job("rendered", "/Game/Person/BP.BP", encoder=encoder)
        self.assertTrue(result["video_verified"])
        self.assertTrue(
            all(
                result["stages"][name]["status"] == "passed"
                for name in (
                    "native_execution",
                    "validate_movie",
                    "encode_video",
                    "verify_encoded_video",
                    "verify_sources",
                    "finalize_resources",
                )
            )
        )
        self.assertGreaterEqual(
            result["seconds"] + 0.001,
            sum(stage["seconds"] for stage in result["stages"].values()),
        )

    def test_video_only_run_never_creates_images_or_runs_alias_wide_maintenance(self):
        encoder = self.root / "ffmpeg.exe"
        encoder.write_bytes(b"synthetic executable")

        def encode(command, **kwargs):
            self.assertEqual(Path(command[command.index("-i") + 1]).suffix, ".mov")
            Path(command[-1]).write_bytes(b"synthetic video")
            return SimpleNamespace(returncode=0)

        with (
            patch("ailias.reanimation.pipeline.subprocess.run", side_effect=encode),
            patch(
                "ailias.reanimation.pipeline.verify_video",
                return_value={"audio_decoded": True},
            ),
            patch("ailias.reanimation.pipeline.inspect_frames") as images,
            patch(
                "ailias.resource_management.retention.finalize_major_output"
            ) as maintenance,
        ):
            store, result = self.job("rendered", "/Game/Person/BP.BP", encoder=encoder)
        self.assertTrue(result["video_verified"])
        root = store.artifact_path(result["artifacts"]["video"]).parent
        self.assertFalse((root / "frames").exists())
        self.assertFalse((root / "portrait-source.mov").exists())
        self.assertNotIn("first_frame", result["artifacts"])
        self.assertFalse(inspect_job(store, result["revision"])["integrity_problems"])
        images.assert_not_called()
        maintenance.assert_not_called()

    def test_missing_direct_movie_fails_before_encoding(self):
        encoder = self.root / "ffmpeg.exe"
        encoder.write_bytes(b"synthetic executable")
        with patch("ailias.reanimation.pipeline.subprocess.run") as encode:
            _, result = self.job("missing_movie", "/Game/Person/BP.BP", encoder=encoder)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["video_verified"])
        self.assertIn("direct movie", result["error"]["message"])
        encode.assert_not_called()

    def test_direct_movie_encoder_cannot_hide_extra_frames_by_truncating_the_input(
        self,
    ):
        command = encode_command(
            Path("ffmpeg.exe"),
            self.root / "source.mov",
            self.audio(),
            self.root / "video.mp4",
            ReelPolicy(),
            3,
        )
        self.assertNotIn("-shortest", command)
        self.assertEqual(
            command[command.index("-af") + 1], "apad,atrim=duration=0.100000000"
        )
        self.assertNotIn("-frames:v", command)
        self.assertNotIn("-t", command)
        self.assertNotIn("-framerate", command)
        self.assertEqual(command[command.index("-map") + 1], "0:v:0")

    def test_native_render_timer_stays_open_until_callback_including_delegate_cleanup(
        self,
    ):
        from ailias.reanimation import unreal_worker

        now = [10.0]
        result = {}
        request = {"receipt": str(self.root / "timed-receipt.json")}
        unreal = SimpleNamespace(
            EditorPythonScripting=SimpleNamespace(
                set_keep_python_script_alive=lambda value: None
            )
        )

        def remove(callback):
            now[0] += 2

        executor = SimpleNamespace(
            on_executor_finished_delegate=SimpleNamespace(remove_callable=remove)
        )
        with patch(
            "ailias.common.execution.time.monotonic", side_effect=lambda: now[0]
        ):
            journal = StageJournal(result)
            journal.begin("render_video")
            now[0] = 30.0
            self.assertNotIn("seconds", result["stages"]["render_video"])
            with (
                patch.dict("sys.modules", {"unreal": unreal}),
                patch.multiple(
                    unreal_worker,
                    REQUEST=request,
                    RESULT=result,
                    JOURNAL=journal,
                    KEEP_EDITOR_ALIVE=True,
                ),
            ):
                unreal_worker.finished(executor, True)
        saved = json.loads(Path(request["receipt"]).read_text())
        self.assertEqual(saved["stages"]["render_video"]["seconds"], 22.0)
        self.assertEqual(saved["worker_seconds"], 22.0)
        self.assertEqual(saved["stages"]["render_video"]["status"], "passed")

    def test_native_failed_render_records_failed_timing_before_terminal_receipt(self):
        from ailias.reanimation import unreal_worker

        for success, cleanup_fails in ((False, False), (True, True)):
            with self.subTest(success=success, cleanup=cleanup_fails):
                result = {}
                journal = StageJournal(result)
                journal.begin("render_video")

                def remove(callback):
                    if cleanup_fails:
                        raise RuntimeError("synthetic cleanup failure")

                executor = SimpleNamespace(
                    on_executor_finished_delegate=SimpleNamespace(
                        remove_callable=remove
                    )
                )
                request = {"receipt": str(self.root / "failed-timing.json")}
                with (
                    patch.dict("sys.modules", {"unreal": SimpleNamespace()}),
                    patch.multiple(
                        unreal_worker,
                        REQUEST=request,
                        RESULT=result,
                        JOURNAL=journal,
                        KEEP_EDITOR_ALIVE=True,
                    ),
                ):
                    unreal_worker.finished(executor, success)
                saved = json.loads(Path(request["receipt"]).read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["stages"]["render_video"]["status"], "failed")
                self.assertIn("seconds", saved["stages"]["render_video"])

    def test_native_journal_loader_verifies_source_and_has_no_host_storage_dependency(
        self,
    ):
        import hashlib
        from ailias.common.code_assets import module_source
        from ailias.reanimation.unreal_worker import load_stage_journal

        source = module_source("ailias.common.execution")
        request = {
            "stage_journal": {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        }
        report = {}
        with patch.dict("sys.modules", {"ailias.storage": None}):
            journal = load_stage_journal(request, report)
            self.assertEqual(journal.execute("native_operation", lambda: 5), 5)
        self.assertEqual(report["stages"]["native_operation"]["status"], "passed")
        request["stage_journal"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source changed"):
            load_stage_journal(request, {})

    def test_chroma_backdrop_compensates_manual_exposure_without_adding_red_or_blue(
        self,
    ):
        from ailias.reanimation.unreal_worker import linear_colour

        unreal = SimpleNamespace(LinearColor=lambda *values: values)
        for exposure in (-5, 0, 2):
            with self.subTest(exposure=exposure):
                colour = linear_colour((0, 1, 0), unreal, exposure)
                self.assertEqual(colour, (0, 2 ** (-exposure), 0, 1))
                self.assertEqual(colour[1] * 2**exposure, 1)
        self.assertAlmostEqual(linear_colour((0.5, 0, 0), unreal)[0], 0.2140411405)

    def test_native_export_offset_is_removed_without_changing_head_animation(self):
        from ailias.reanimation.unreal_worker import normalize_subject_root

        class Channel:
            def __init__(self, value):
                self.value = value
                self.keys = 0

            def get_num_keys(self):
                return self.keys

            def set_default(self, value):
                self.value = value

        class TransformTrack:
            def get_sections(self):
                return [SimpleNamespace(get_channels_by_type=lambda _: channels)]

        actor_class = object()
        channels = [Channel(value) for value in (50, 0, 0, 0, 0, 90, 1, 1, 1)]
        head_keys = [1, 2, 3]
        binding = SimpleNamespace(
            get_object_template=lambda: SimpleNamespace(get_class=lambda: actor_class),
            get_tracks=lambda: [TransformTrack(), SimpleNamespace(keys=head_keys)],
        )
        sequence = SimpleNamespace(get_bindings=lambda: [binding])
        unreal = SimpleNamespace(
            MovieScene3DTransformTrack=TransformTrack,
            MovieSceneScriptingDoubleChannel=Channel,
        )
        normalize_subject_root(sequence, actor_class, unreal)
        self.assertEqual(
            [channel.value for channel in channels], [0, 0, 0, 0, 0, 0, 1, 1, 1]
        )
        self.assertEqual(head_keys, [1, 2, 3])
        normalize_subject_root(sequence, actor_class, unreal, (40, 60, 0), 30)
        self.assertEqual(
            [channel.value for channel in channels], [40, 60, 0, 0, 0, 30, 1, 1, 1]
        )
        channels[0].value, channels[-1].keys = 50, 1
        with self.assertRaisesRegex(ValueError, "animated MetaHuman root"):
            normalize_subject_root(sequence, actor_class, unreal)
        self.assertEqual(channels[0].value, 50)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            normalize_subject_root(
                SimpleNamespace(get_bindings=lambda: []), actor_class, unreal
            )


if __name__ == "__main__":
    unittest.main()
