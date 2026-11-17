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

from ailias.reanimation.media import encode_command, inspect_audio, inspect_frames, verify_video
from ailias.reanimation.pipeline import REQUIRED_PLUGINS, create_reel, inspect_job
from ailias.reanimation.settings import ReelPolicy
from ailias.storage import Store, atomic_json


class ReelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def audio(self, silent=False):
        path = self.root / "voice.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            stream.writeframes(array.array("h", [0 if silent else 1000]*2400).tobytes())
        return path

    def cli_args(self, extra=()):
        from ailias.cli import parser
        return parser().parse_args([
            "reanimation", "run", "subject", "--audio", str(self.root / "voice.wav"),
            "--editor", str(self.root / "UnrealEditor-Cmd.exe"),
            "--project", str(self.root / "Review.uproject"), *extra])

    def test_cli_resolution_override_preserves_lighting_and_camera_without_changing_preset(self):
        from ailias.reanimation.cli import execute

        config = self.root / "custom.json"
        atomic_json(config, {"width": 720, "height": 1280, "key_lumens": 750})
        framing = ReelPolicy().framing()
        for extra, dimensions, lumens in (
                ([], (1080, 1920), 900),
                (["--resolution", "540x960"], (540, 960), 900),
                (["--reel-config", str(config)], (720, 1280), 750),
                (["--reel-config", str(config), "--resolution", "540X960"], (540, 960), 750)):
            with self.subTest(extra=extra), patch(
                    "ailias.reanimation.pipeline.create_reel",
                    return_value={"status": "audio_solved_awaiting_rigged_metahuman"}) as native:
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
        args = self.cli_args([
            "--blueprint", model, "--audio", str(audio),
            "--character-position", "40", "60", "0", "--character-yaw", "30",
            "--screen-position", ".45", ".52"])
        with patch("ailias.reanimation.pipeline.create_reel",
                   return_value={"status": "video_rendered_awaiting_visual_review"}) as native:
            execute(args, self.root / "data")
        self.assertEqual(native.call_args.args[1], audio)
        self.assertEqual(native.call_args.kwargs["blueprint"], model)
        policy = native.call_args.args[4]
        self.assertEqual(policy.character_location_cm, [40, 60, 0])
        self.assertEqual(policy.character_yaw_degrees, 30)
        self.assertEqual(policy.subject_screen_position, [.45, .52])

    def test_camera_shift_maps_normalized_screen_placement_in_rotated_rig_axes(self):
        policy = ReelPolicy(character_yaw_degrees=90, subject_screen_position=(.25, .75))
        placement = policy.placement()
        for actual, expected in zip(placement["front"], (-1, 0, 0)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(placement["camera_target_shift_cm"], (0, 7.3125, 13)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(ReelPolicy().placement()["camera_target_shift_cm"], (0, 0, 0))

    def test_portrait_lens_geometry_preserves_face_field_of_view(self):
        policy = ReelPolicy()
        framing = policy.framing()
        height = 2*framing["camera_distance_cm"]*math.tan(
            math.radians(framing["vertical_fov_degrees"])/2)
        self.assertAlmostEqual(height, 52)
        self.assertAlmostEqual(framing["visible_width_cm"], 29.25)
        self.assertAlmostEqual(framing["camera_distance_cm"], 122.7777778)
        for values in ({"width": 1920}, {"fps": True}, {"mood_intensity": float("nan")},
                       {"background_srgb": [1, 2, 3]}, {"encoder": "auto"},
                       {"exposure_compensation_ev": -13}, {"exposure_compensation_ev": True},
                       {"light_size_cm": 0}, {"light_size_cm": float("inf")},
                       {"character_location_cm": [0, 0]}, {"character_location_cm": [0, True, 0]},
                       {"subject_screen_position": [1.1, .5]}, {"character_yaw_degrees": 181}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ReelPolicy(**values)

    def test_narration_rejects_silence_and_excess_duration(self):
        self.assertAlmostEqual(inspect_audio(self.audio())["duration_seconds"], .1)
        with self.assertRaisesRegex(ValueError, "Silent"):
            inspect_audio(self.audio(True))
        with self.assertRaisesRegex(ValueError, "duration"):
            inspect_audio(self.audio(), .05)

    def test_frames_require_contiguous_complete_portrait_dimensions(self):
        policy = ReelPolicy(width=90, height=160)
        for index in (0, 1):
            Image.new("RGB", (90, 160)).save(self.root / f"frame.{index:06d}.png")
        self.assertEqual(len(inspect_frames(self.root, policy, 2)), 2)
        (self.root / "frame.000000.png").rename(self.root / "frame.000002.png")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            inspect_frames(self.root, policy, 2)

    def test_gpu_encode_is_explicit_and_never_uses_shortest_or_overwrites(self):
        command = encode_command(Path("ffmpeg.exe"), self.root, self.audio(),
                                 self.root / "reel.mp4", ReelPolicy(), 3)
        self.assertIn("h264_nvenc", command)
        self.assertIn("-n", command)
        self.assertNotIn("-shortest", command)
        self.assertEqual(command[command.index("-frames:v")+1], "3")

    def job(self, outcome, blueprint=None):
        project = self.root / "Review.uproject"
        atomic_json(project, {"Plugins": [{"Name": name, "Enabled": True} for name in REQUIRED_PLUGINS]})
        editor = self.root / "UnrealEditor-Cmd.exe"
        editor.write_bytes(b"fake")
        store = Store(self.root / "data", "subject")

        def launch(name, operation, args, directory, environment):
            request = json.loads(Path(environment["AILIAS_REANIMATION_REQUEST"]).read_text())
            performance = request["destination"] + "/Performance.Performance"
            if outcome != "missing":
                package = project.parent / "Content" / Path(*request["destination"].split("/")[2:])
                package.mkdir(parents=True)
                (package / "Performance.uasset").write_bytes(b"fake native package")
                (package / "Narration.uasset").write_bytes(b"fake imported sound")
            atomic_json(Path(request["receipt"]), {
                "status": "audio_solved_awaiting_rigged_metahuman", "performance": performance,
                "sound": request["destination"] + "/Narration.Narration",
                "processed_frames": 0 if outcome == "empty" else 3})
            if outcome == "crash":
                raise ValueError("native exit_3221225477")
            return {"seconds": 0}

        result = create_reel(store, self.audio(), editor, project, ReelPolicy(),
                             blueprint=blueprint, tools=SimpleNamespace(paths={}, run=launch))
        return store, result

    def test_audio_solve_without_rig_never_claims_video(self):
        store, result = self.job("saved")
        self.assertEqual(result["status"], "audio_solved_awaiting_rigged_metahuman")
        self.assertFalse(result["video_verified"])
        self.assertTrue(store.verify(result["artifacts"]["performance"]))
        self.assertEqual(inspect_job(store, result["revision"])["integrity_problems"], [])

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
                (3, 100000, 106667, True), (2, 100000, 106667, False),
                (3, 500000, 106667, False), (3, 100000, 10000, False)):
            video = SimpleNamespace(returncode=0, stdout=f"frame={count}\nout_time_us={video_time}\n")
            audio = SimpleNamespace(returncode=0, stdout=f"out_time_us={audio_time}\n")
            with self.subTest(count=count, video_time=video_time, audio_time=audio_time):
                with patch("ailias.reanimation.media.subprocess.run", side_effect=[metadata, video, audio]):
                    if valid:
                        self.assertTrue(verify_video(Path("ffmpeg.exe"), Path("reel.mp4"),
                                                     ReelPolicy(), 3)["audio_decoded"])
                    else:
                        with self.assertRaises(ValueError):
                            verify_video(Path("ffmpeg.exe"), Path("reel.mp4"), ReelPolicy(), 3)

    def test_render_completion_yields_shutdown_to_startup_script_owner(self):
        from ailias.reanimation import unreal_worker
        keepalive = []
        unreal = SimpleNamespace(EditorPythonScripting=SimpleNamespace(
            set_keep_python_script_alive=keepalive.append))
        receipt = self.root / "receipt.json"
        with patch.dict("sys.modules", {"unreal": unreal}), \
                patch.object(unreal_worker, "REQUEST", {"receipt": str(receipt)}), \
                patch.object(unreal_worker, "RESULT", {}):
            unreal_worker.finished(None, True)
        self.assertEqual(keepalive, [False])
        self.assertEqual(json.loads(receipt.read_text())["status"], "frames_rendered")

    def test_chroma_backdrop_compensates_manual_exposure_without_adding_red_or_blue(self):
        from ailias.reanimation.unreal_worker import linear_colour

        unreal = SimpleNamespace(LinearColor=lambda *values: values)
        for exposure in (-5, 0, 2):
            with self.subTest(exposure=exposure):
                colour = linear_colour((0, 1, 0), unreal, exposure)
                self.assertEqual(colour, (0, 2**(-exposure), 0, 1))
                self.assertEqual(colour[1]*2**exposure, 1)
        self.assertAlmostEqual(linear_colour((.5, 0, 0), unreal)[0], .2140411405)

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
            get_tracks=lambda: [TransformTrack(), SimpleNamespace(keys=head_keys)])
        sequence = SimpleNamespace(get_bindings=lambda: [binding])
        unreal = SimpleNamespace(MovieScene3DTransformTrack=TransformTrack,
                                 MovieSceneScriptingDoubleChannel=Channel)
        normalize_subject_root(sequence, actor_class, unreal)
        self.assertEqual([channel.value for channel in channels], [0, 0, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(head_keys, [1, 2, 3])
        normalize_subject_root(sequence, actor_class, unreal, (40, 60, 0), 30)
        self.assertEqual([channel.value for channel in channels], [40, 60, 0, 0, 0, 30, 1, 1, 1])
        channels[0].value, channels[-1].keys = 50, 1
        with self.assertRaisesRegex(ValueError, "animated MetaHuman root"):
            normalize_subject_root(sequence, actor_class, unreal)
        self.assertEqual(channels[0].value, 50)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            normalize_subject_root(SimpleNamespace(get_bindings=lambda: []), actor_class, unreal)


if __name__ == "__main__":
    unittest.main()
