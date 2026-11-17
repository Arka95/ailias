"""Exercise neutral review publication and narration-free worker dispatch with fakes."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from ailias.facefitting.native_features import character_package
from ailias.facefitting.native_review import render_review
from ailias.reanimation import unreal_worker
from ailias.reanimation.settings import ReelPolicy
from ailias.storage import Store, atomic_json


class NativeReviewTests(unittest.TestCase):
    def review(self, outcome):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = Store(root / "data", "subject")
        project = root / "Project" / "Review.uproject"
        project.parent.mkdir()
        project.write_text("{}")
        package = project.parent / "Content" / "BP.uasset"
        package.parent.mkdir()
        package.write_bytes(b"synthetic runtime actor")
        source = store.resolve("facefitting\\features\\fitted\\manifest.json")
        feature = {
            "status": "features_reloaded_awaiting_visual_review", "fresh_editor_reload_verified": True,
            "blueprint": "/Game/BP.BP", "readiness": {"project": str(project)},
            "input_dependencies": [], "artifacts": {"actor": store.reference_artifact(package)},
        }
        diagnostic = outcome.startswith("diagnostic-")
        if diagnostic or outcome == "failed-feature":
            feature.update(status="failed", fresh_editor_reload_verified=False)
            blueprint = feature.pop("blueprint")
            operation = "build" if outcome == "diagnostic-build" else "apply"
            feature[operation] = {"runtime": {"blueprint": blueprint}}
        atomic_json(source, feature)
        if outcome == "changed-source":
            package.write_bytes(b"changed runtime")
        policy = ReelPolicy(width=36, height=64)
        calls = []

        def launch(name, operation, args, directory, environment):
            request = json.loads(Path(environment["AILIAS_REANIMATION_REQUEST"]).read_text())
            calls.append(request)
            self.assertTrue(request["neutral_review"])
            self.assertNotIn("audio", request)
            self.assertEqual(request["frame_count"], 1)
            result = {"status": "frames_rendered", "render_success": True, "neutral_review": True}
            for key, name in (("sequence", "Sequence"), ("map", "Stage"), ("material", "Backdrop")):
                asset = request["destination"] + "/" + name
                result[key] = asset
                path = character_package(project, asset)
                if key == "map":
                    path = path.with_suffix(".umap")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic native package")
            if outcome != "missing-frame":
                Image.new("RGB", (36, 64)).save(Path(request["frames"]) / "frame.000000.png")
            if outcome == "animated":
                result["neutral_review"] = False
            Path(request["receipt"]).write_text(json.dumps(result))
            return {"seconds": 0}

        with patch("ailias.facefitting.native_review.doctor", return_value={"blockers": []}):
            report = render_review(store, "fitted", root / "Editor.exe", project, policy,
                                   SimpleNamespace(paths={}, run=launch), diagnostic=diagnostic)
        persisted = store.resolve("facefitting\\native-reviews\\" + report["revision"] + "\\manifest.json")
        self.assertEqual(json.loads(persisted.read_text())["status"], report["status"])
        return store, report, calls

    def test_neutral_review_requires_real_frame_and_retains_native_and_source_evidence(self):
        store, report, calls = self.review("saved")
        self.assertEqual(report["status"], "native_neutral_rendered_awaiting_visual_review")
        self.assertTrue(store.verify(report["artifacts"]["image"]))
        self.assertFalse(report["likeness_verified"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(store.verify(item) for item in report["input_dependencies"]))

    def test_missing_frame_or_animated_receipt_never_publishes_neutral_success(self):
        for outcome in ("missing-frame", "animated"):
            with self.subTest(outcome=outcome):
                _, report, _ = self.review(outcome)
                self.assertEqual(report["status"], "failed")
                self.assertNotIn("image", report["artifacts"])

    def test_changed_runtime_is_rejected_before_rendering(self):
        with self.assertRaisesRegex(ValueError, "dependencies changed"):
            self.review("changed-source")

    def test_failed_feature_requires_explicit_diagnostic_mode_for_either_build_layout(self):
        with self.assertRaisesRegex(ValueError, "successfully reloaded"):
            self.review("failed-feature")
        for outcome in ("diagnostic-build", "diagnostic-apply"):
            with self.subTest(outcome=outcome):
                _, report, calls = self.review(outcome)
                self.assertEqual(report["status"], "native_neutral_rendered_awaiting_visual_review")
                self.assertTrue(report["diagnostic"])
                self.assertEqual(report["source_feature_status"], "failed")
                self.assertFalse(report["likeness_verified"])
                self.assertEqual(len(calls), 1)

    def test_neutral_dispatch_skips_audio_but_normal_reanimation_still_solves(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        fake = SimpleNamespace(SystemLibrary=SimpleNamespace(get_engine_version=lambda: "5.8.4-fixture"))
        for neutral in (False, True):
            request = root / "request.json"
            request.write_text(json.dumps({
                "blueprint": "/Game/Actor.Actor", "neutral_review": neutral,
                "receipt": str(root / "receipt.json"),
            }))
            sequence, performance = object(), object()
            with self.subTest(neutral=neutral), patch.dict(sys.modules, {"unreal": fake}), \
                    patch.dict(os.environ, {"AILIAS_REANIMATION_REQUEST": str(request)}), \
                    patch.object(unreal_worker, "rig_class"), \
                    patch.object(unreal_worker, "solve", return_value=(performance, {})) as solve, \
                    patch.object(unreal_worker, "stage", return_value=(sequence, {})) as stage, \
                    patch.object(unreal_worker, "render") as render:
                unreal_worker.main()
                self.assertEqual(solve.call_count, 0 if neutral else 1)
                self.assertIs(stage.call_args.args[1], None if neutral else performance)
                self.assertIs(render.call_args.args[1], sequence)


if __name__ == "__main__":
    unittest.main()
