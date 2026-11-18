"""Verify appearance-to-Unreal persistence/orientation contracts without launching an engine."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ailias.facefitting.unreal.static_mesh_import import import_appearance
from ailias.storage import Store, atomic_json


class UnrealAppearanceTests(unittest.TestCase):
    def imported(self, orientation_correct):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = Store(root / "data", "subject")
        source_root = store.resolve("facefitting\\appearance\\revision")
        source_root.mkdir(parents=True)
        vertices = np.array([[-.1, .1, -.1], [.1, .4, .15]])
        np.savez(source_root / "geometry.npz", vertices=vertices)
        mesh = source_root / "head.obj"
        mesh.write_text("synthetic head, never imported")
        source = {"status": "awaiting_appearance_review", "integrity_problems": [], "artifacts": {
            "head_only_mesh": store.reference_artifact(mesh),
            "geometry": store.reference_artifact(source_root / "geometry.npz")}}
        atomic_json(source_root / "manifest.json", source)
        project = root / "Project" / "Review.uproject"
        project.parent.mkdir()
        project.write_text("{}")
        probe = {"blockers": [], "editor": str(root / "UnrealEditor-Cmd.exe"), "project": str(project)}

        def launch(name, operation, args, directory, environment):
            request = json.loads(Path(environment["AILIAS_UNREAL_REQUEST"]).read_text())
            destination = request["destination"]
            points = vertices[:, [2, 0, 1]]*[100, -100, 100] if orientation_correct else vertices
            packages = [destination+"/Head.Head", destination+"/TEX_skin-basecolor.TEX_skin-basecolor"]
            for package in packages:
                relative = package.removeprefix("/Game/").split(".")[0]
                saved = project.parent / "Content" / Path(*relative.split("/"))
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.with_suffix(".uasset").write_bytes(b"synthetic package, never loaded")
            atomic_json(Path(request["receipt"]), {"status": "mesh_imported", "assets": packages,
                "mesh_geometry": [{"bounds_min_xyz": points.min(0).tolist(),
                                   "bounds_max_xyz": points.max(0).tolist()}]})
            return {"seconds": 0}

        with patch("ailias.facefitting.unreal.static_mesh_import.inspect_head_appearance", return_value=source), \
                patch("ailias.facefitting.unreal.static_mesh_import.readiness", return_value=probe):
            result = import_appearance(store, "revision", Path(probe["editor"]), project,
                                       SimpleNamespace(paths={}, run=launch))
        return store, result

    def test_import_hashes_real_packages_with_hyphens_without_claiming_metahuman(self):
        store, result = self.imported(True)
        self.assertEqual(result["status"], "static_head_imported_for_review")
        self.assertTrue(result["upright_orientation_verified"])
        self.assertTrue(store.verify(result["artifacts"]["asset_1"]))
        self.assertFalse(result["metahuman_created"])
        self.assertFalse(result["rig_verified"])

    def test_wrong_import_axis_is_not_published_as_a_successful_handoff(self):
        _, result = self.imported(False)
        self.assertEqual(result["status"], "failed")
        self.assertIn("orientation/scale", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
