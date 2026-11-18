"""Editor callbacks and failure receipts without starting Unreal or rendering hardware."""

import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from ailias.facefitting.unreal.character_portrait_worker import calculate_portrait_framing, main
from ailias.storage import atomic_json


class NativePreviewTests(unittest.TestCase):
    def execute(self, *, export_fails=False, missing_grooms=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        request = root / "request.json"
        receipt, output = root / "receipt.json", root / "new.png"
        atomic_json(request, {"operation": "preview", "receipt": str(receipt),
                             "output": str(output), "character": "/Game/Current/Character",
                             "destination": "/Game/Current/Preview", "warmup_seconds": 20,
                             "lighting": {"key_lumens": 900, "fill_ratio": .65, "light_size_cm": 200,
                                          "temperature_kelvin": 5600, "exposure_compensation_ev": -5.5}})

        class Vector:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

            def __add__(self, other):
                return Vector(self.x+other.x, self.y+other.y, self.z+other.z)

        class Character:
            pass

        character = Character()
        face, groom = Mock(), Mock()
        face.get_name.return_value = "Face"
        groom.get_name.return_value = "Hair"
        groom_asset = Mock()
        groom_asset.get_path_name.return_value = "/MetaHumanCharacter/Grooms/CurrentHair"
        groom.get_editor_property.return_value = None if missing_grooms else groom_asset
        subject = Mock()
        subject.get_all_child_actors.return_value = []
        subject.get_components_by_class.side_effect = lambda cls: [face] if cls == "skeletal" else [groom]
        subsystem, actors, levels, world = Mock(), Mock(), Mock(), Mock()
        subsystem.get_face_landmarks.return_value = [
            Vector(-8, 0, 142), Vector(9, 0, 168), Vector(0, 16, 155)]
        subsystem.spawn_meta_human_actor.return_value = subject
        capture_actor = Mock(spec=["get_editor_property", "get_actor_location", "set_actor_rotation"])
        capture = Mock()
        capture_actor.get_editor_property.return_value = capture
        capture_actor.get_actor_location.return_value = Vector(0, 75, 0)
        capture.post_process_settings = SimpleNamespace()
        actors.spawn_actor_from_class.side_effect = lambda cls, location: (
            capture_actor if cls == "capture" else Mock())
        rendering = Mock()

        def export(*args):
            if export_fails:
                raise OSError("preview disk unavailable")
            Image.new("RGB", (540, 960), (100, 80, 60)).save(output)

        rendering.export_render_target.side_effect = export
        unreal = SimpleNamespace(
            get_editor_subsystem=lambda cls: {"native": subsystem, "actors": actors,
                                             "levels": levels, "world": world}[cls],
            MetaHumanCharacterEditorSubsystem="native", EditorActorSubsystem="actors",
            LevelEditorSubsystem="levels", UnrealEditorSubsystem="world",
            MetaHumanCharacter=Character, load_asset=lambda name: character,
            SkeletalMeshComponent="skeletal", GroomComponent="groom",
            RectLight="light", SceneCapture2D="capture", Vector=Vector,
            LinearColor=lambda *args: args, MathLibrary=Mock(),
            SystemLibrary=SimpleNamespace(get_component_bounds=lambda component: (
                Vector(0, 0, 150), Vector(12, 12, 20), 30)),
            LightUnits=SimpleNamespace(LUMENS="lumens"), RenderingLibrary=rendering,
            TextureRenderTargetFormat=SimpleNamespace(RTF_RGBA8="rgba8"),
            SceneCaptureSource=SimpleNamespace(SCS_FINAL_COLOR_LDR="ldr"),
            AutoExposureMethod=SimpleNamespace(AEM_MANUAL="manual"),
            EditorPythonScripting=Mock(), register_slate_post_tick_callback=Mock(return_value="tick"),
            unregister_slate_post_tick_callback=Mock())
        with patch.dict("sys.modules", {"unreal": unreal}), \
                patch.dict(os.environ, {"AILIAS_FRESH_VIDEO_REQUEST": str(request)}), \
                patch("ailias.facefitting.unreal.character_portrait_worker.time.monotonic", side_effect=[0, 21, 23]):
            if missing_grooms:
                with self.assertRaisesRegex(RuntimeError, "no actual groom assets"):
                    main()
            else:
                main()
                callback = unreal.register_slate_post_tick_callback.call_args.args[0]
                callback(.016)
                callback(.016)
        return json.loads(receipt.read_text()), unreal, subsystem, output

    def test_preview_keeps_editor_ticks_alive_then_writes_a_new_image_and_releases_owned_state(self):
        result, unreal, subsystem, output = self.execute()
        self.assertEqual(result["status"], "native_editor_preview_rendered")
        self.assertTrue(output.is_file())
        self.assertFalse(result["assembled"])
        subsystem.spawn_meta_human_actor.assert_called_once_with(
            subsystem.try_add_object_to_edit.call_args.args[0], keep_transient=True)
        unreal.unregister_slate_post_tick_callback.assert_called_once_with("tick")
        subsystem.remove_object_to_edit.assert_called_once()
        unreal.EditorPythonScripting.set_keep_python_script_alive.assert_called_with(False)

    def test_export_failure_is_persisted_and_does_not_leave_the_editor_script_alive(self):
        result, unreal, subsystem, output = self.execute(export_fails=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "preview disk unavailable")
        self.assertFalse(output.exists())
        subsystem.remove_object_to_edit.assert_called_once()
        unreal.EditorPythonScripting.set_keep_python_script_alive.assert_called_with(False)

    def test_an_actor_without_actual_groom_assets_is_not_a_completed_preview(self):
        result, _, subsystem, output = self.execute(missing_grooms=True)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(output.exists())
        subsystem.remove_object_to_edit.assert_called_once()

    def test_framing_uses_head_controls_and_keeps_a_portrait_closeup(self):
        points = [SimpleNamespace(x=-8, y=0, z=142), SimpleNamespace(x=9, y=16, z=168),
                  SimpleNamespace(x=0, y=8, z=155)]
        result = calculate_portrait_framing(points)
        self.assertEqual(result["center_cm"], [.5, 8, 155])
        self.assertAlmostEqual(result["vertical_span_cm"], 35.1)
        actual_vertical_span = 2*75*math.tan(math.radians(result["fov_degrees"])/2)*960/540
        self.assertAlmostEqual(actual_vertical_span, result["vertical_span_cm"])

    def test_nonfinite_or_degenerate_face_bounds_fail_before_rendering(self):
        for points in ([SimpleNamespace(x=0, y=0, z=0)]*3,
                       [SimpleNamespace(x=float("nan"), y=0, z=0)]*3):
            with self.assertRaises(ValueError):
                calculate_portrait_framing(points)
