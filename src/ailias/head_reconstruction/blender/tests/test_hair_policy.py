"""A hair-free scene must not depend on inferred roots or touch Blender geometry."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class BlenderHairPolicyTests(unittest.TestCase):
    def test_zero_strands_skips_cap_curves_and_root_inference(self):
        path = Path(__file__).parents[1] / "appearance_scene_worker.py"
        spec = importlib.util.spec_from_file_location("appearance_worker_test", path)
        worker = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "bpy": SimpleNamespace(),
            "mathutils": SimpleNamespace(Matrix=None, Vector=None),
        }):
            spec.loader.exec_module(worker)
        objects, report = worker.add_hair(None, None, None, None, None,
                                          {"policy": {"hair_strands": 0}})
        self.assertEqual(objects, [])
        self.assertEqual(report, {"status": "disabled", "strands": 0})
