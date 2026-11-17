"""Exercise real MCP request construction with a fake SDK; never start speech or native tools."""

import array
import contextlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ailias.reanimation.narration import generate_narration
from ailias.storage import Store, atomic_json


class NarrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / "data", "subject")
        self.script = self.root / "script.json"
        atomic_json(self.script, {"title": "Reviewed", "beats": [{"text": "Selected voice test."}]})
        self.config = self.root / "server.local.json"
        atomic_json(self.config, {"command": "existing-python.exe", "args": ["-m", "yapit.mcp"],
                                 "cwd": str(self.root), "env": {"YAPIT_DATA_DIR": "existing-voices"}})
        self.wav = self.root / "generated.wav"
        with wave.open(str(self.wav), "wb") as stream:
            stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            stream.writeframes(array.array("h", [1000]*2400).tobytes())
        self.provenance = self.root / "provenance.json"
        atomic_json(self.provenance, {"voice": "selected", "backend": "pocket", "device": "cpu"})
        self.output = {"wav": str(self.wav), "provenance": str(self.provenance)}

    @contextlib.contextmanager
    def sdk(self, *, error=False, failure=None, text_only=False):
        response = SimpleNamespace(
            is_error=error, structured_content=None if text_only else self.output,
            content=[SimpleNamespace(type="text", text=json.dumps(self.output))],
            model_dump=lambda **_: {"is_error": error, "result": self.output})
        session = SimpleNamespace(initialize=AsyncMock(),
                                  call_tool=AsyncMock(return_value=response, side_effect=failure))
        captured = SimpleNamespace(session=session, launches=[])

        @contextlib.asynccontextmanager
        async def stdio(parameters, **kwargs):
            captured.launches.append(parameters)
            yield "reader", "writer"

        @contextlib.asynccontextmanager
        async def client(*args, **kwargs):
            yield session

        with patch.dict("sys.modules", {
                "mcp": SimpleNamespace(ClientSession=client, StdioServerParameters=lambda **kw: SimpleNamespace(**kw)),
                "mcp.client.stdio": SimpleNamespace(stdio_client=stdio)}):
            yield captured

    def generate(self, **kwargs):
        return generate_narration(self.store, "selected", self.script, self.config, **kwargs)

    def test_selected_profile_and_reviewed_script_reach_actual_mcp_call_shape(self):
        for text_only in (False, True):
            with self.subTest(text_only=text_only), self.sdk(text_only=text_only) as sdk:
                narration = self.generate()
                name, request = sdk.session.call_tool.call_args.args
                self.assertEqual(name, "generate_audio")
                self.assertEqual(request, {"voice": "selected", "style": "default",
                                          "script": json.loads(self.script.read_text())})
                parameters = sdk.launches[0]
                self.assertEqual(parameters.command, "existing-python.exe")
                self.assertEqual(parameters.args, ["-m", "yapit.mcp"])
                self.assertEqual(parameters.env["YAPIT_DATA_DIR"], "existing-voices")
                self.assertEqual(parameters.env["YAPIT_SPEECH_CACHE"], "0")
                self.assertTrue(Path(parameters.env["YAPIT_WORKSPACE"]).is_relative_to(self.store.root))
                self.assertNotEqual(narration["audio"], self.wav)
                self.assertEqual(narration["audio"].read_bytes(), self.wav.read_bytes())
                for reference in narration["artifacts"].values():
                    self.assertTrue(self.store.verify(reference))

    def test_each_call_is_fresh_and_explicit_duration_and_style_are_forwarded(self):
        with self.sdk() as sdk:
            first = self.generate(seconds=10, style="energetic")
            second = self.generate(seconds=10, style="energetic")
            self.assertEqual(sdk.session.call_tool.await_count, 2)
            self.assertEqual(sdk.session.call_tool.call_args.args[1]["seconds"], 10)
            self.assertEqual(sdk.session.call_tool.call_args.args[1]["style"], "energetic")
            self.assertNotEqual(first["audio"], second["audio"])

    def test_wrong_profile_and_tool_failure_never_publish_audio(self):
        for kind in ("wrong-voice", "tool-error", "timeout"):
            with self.subTest(kind=kind):
                atomic_json(self.provenance, {"voice": "other" if kind == "wrong-voice" else "selected"})
                with self.sdk(error=kind == "tool-error",
                              failure=TimeoutError() if kind == "timeout" else None):
                    with self.assertRaisesRegex(ValueError, "Yapit narration failed"):
                        self.generate()
        manifests = list(self.store.root.glob("reanimation/narration/*/manifest.json"))
        self.assertEqual(len(manifests), 3)
        for manifest in manifests:
            self.assertEqual(json.loads(manifest.read_text())["status"], "failed")

    def test_generated_audio_must_pass_reel_duration_limit(self):
        with self.sdk(), self.assertRaisesRegex(ValueError, "Yapit narration failed"):
            self.generate(max_audio_seconds=.05)
        manifest = next(self.store.root.glob("reanimation/narration/*/manifest.json"))
        self.assertEqual(json.loads(manifest.read_text())["status"], "failed")

    def test_invalid_duration_is_rejected_before_launch_or_job_creation(self):
        with self.sdk() as sdk:
            for seconds in (float("nan"), 0, 181, True):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    self.generate(seconds=seconds)
            sdk.session.call_tool.assert_not_called()
        self.assertFalse(self.store.root.exists())

    def voice_args(self, extra=()):
        from ailias.cli import parser
        return parser().parse_args([
            "reanimation", "run", "--alias", "subject", "--voice", "selected",
            "--script", str(self.script), "--yapit-config", str(self.config),
            "--editor", str(self.root / "editor.exe"), "--project", str(self.root / "Review.uproject"),
            "--resolution", "540x960", *extra])

    def test_cli_selected_voice_becomes_render_audio_and_retains_provenance(self):
        from ailias.reanimation.cli import execute

        with self.sdk(), patch("ailias.reanimation.pipeline.doctor", return_value={"blockers": []}), patch(
                "ailias.reanimation.pipeline.create_reel",
                return_value={"status": "audio_solved_awaiting_rigged_metahuman"}) as native:
            _, code = execute(self.voice_args(), self.root / "data")
            self.assertEqual(code, 0)
            self.assertTrue(native.call_args.args[1].is_file())
            self.assertNotEqual(native.call_args.args[1], self.wav)
            self.assertEqual(native.call_args.kwargs["narration"]["voice"], "selected")
            self.assertEqual(native.call_args.args[4].width, 540)

    def test_synthesis_failure_and_preflight_blockers_never_launch_renderer(self):
        from ailias.reanimation.cli import execute

        with self.sdk(error=True) as sdk, patch("ailias.reanimation.pipeline.create_reel") as native:
            with patch("ailias.reanimation.pipeline.doctor", return_value={"blockers": ["missing editor"]}):
                self.assertEqual(execute(self.voice_args(), self.root / "data")[1], 2)
                sdk.session.call_tool.assert_not_called()
            with patch("ailias.reanimation.pipeline.doctor", return_value={"blockers": []}):
                with self.assertRaises(ValueError):
                    execute(self.voice_args(), self.root / "data")
            native.assert_not_called()
        self.assertFalse((self.store.root / ".lock").exists())

    def test_voice_and_wav_are_exclusive_and_script_is_required_only_for_synthesis(self):
        from ailias.reanimation.cli import execute

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.voice_args(["--audio", str(self.wav)])
        args = self.voice_args()
        args.script = None
        with self.assertRaisesRegex(ValueError, "--voice requires --script"):
            execute(args, self.root / "data")
        args.voice, args.audio, args.script = None, self.wav, self.script
        with self.assertRaisesRegex(ValueError, "apply only to --voice"):
            execute(args, self.root / "data")
        for voice in ("", " "):
            args = self.voice_args()
            args.voice = voice
            with self.subTest(voice=voice), self.assertRaisesRegex(ValueError, "existing Yapit voice"):
                execute(args, self.root / "data")
        self.assertFalse(self.store.root.exists())


if __name__ == "__main__":
    unittest.main()
