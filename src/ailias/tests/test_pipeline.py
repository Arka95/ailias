import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from ailias.cli import main
from ailias.intake import VIEWS, analyze_image, ingest, review_photo, validate
from ailias.settings import Policy, load_settings
from ailias.storage import Store
from ailias.workflow import attach, baseline, status


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "data", "subject-01")
        self.store.initialize()
        self.policy = Policy(min_short_side=8, min_edge_energy=0,
                             max_dark_fraction=1, max_bright_fraction=1)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        for index in range(len(VIEWS)):
            self.image(self.photos / f"{index}.png", color=30 + 18 * index)
        self.mesh = self.root / "head.obj"
        self.mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
        self.texture = self.root / "texture.png"
        self.image(self.texture, 128)

    def image(self, path, color):
        Image.new("RGB", (32, 32), (color, color, color)).save(path)

    def ready_photos(self):
        result = ingest(self.store, self.photos, self.policy)
        self.ids = [photo["id"] for photo in result["photos"]]
        for photo_id, (yaw, pitch) in zip(self.ids, VIEWS.values()):
            review_photo(self.store, photo_id[:12], yaw, pitch, True, True, True)
        return validate(self.store, self.policy)

    def reconstruction(self):
        cameras = {
            photo_id: {
                "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0],
                                    [0, 0, 1, 0], [0, 0, 0, 1]],
                "intrinsics": [[100, 0, 16], [0, 100, 16], [0, 0, 1]],
                "reprojection_error_px": 0.8,
            } for photo_id in self.ids
        }
        self.cameras = self.root / "cameras.json"
        self.cameras.write_text(json.dumps(cameras), encoding="utf-8")
        return attach(self.store, self.policy, "reconstruction",
                      {"mesh": self.mesh, "texture": self.texture, "cameras": self.cameras},
                      "Synthetic manual receipt; no real reconstruction executed")

    def metahuman(self):
        self.ready_photos()
        self.reconstruction()
        attach(self.store, self.policy, "cleanup",
               {"mesh": self.mesh, "texture": self.texture}, "Fixture cleanup")
        attach(self.store, self.policy, "metahuman",
               {"evidence": self.texture}, "Fixture screenshot only")

    def test_quality_gate_reports_resolution_blur_and_exposure(self):
        path = self.root / "dark.png"
        self.image(path, 0)
        result = analyze_image(path, Policy(min_short_side=64))
        self.assertEqual(set(result["reasons"]),
                         {"resolution_below_minimum", "low_detail_or_blur", "underexposed"})
        self.image(path, 255)
        self.assertIn("overexposed", analyze_image(path, Policy())["reasons"])

    def test_intake_handles_corruption_and_deduplicates_content(self):
        (self.photos / "bad.jpg").write_bytes(b"not an image")
        (self.photos / "copy.png").write_bytes((self.photos / "0.png").read_bytes())
        result = ingest(self.store, self.photos, self.policy)
        self.assertEqual(result["total"], len(VIEWS))
        self.assertTrue(any("error" in record for record in result["photos"]))
        self.assertEqual(ingest(self.store, self.photos, self.policy)["total"], len(VIEWS))
        self.assertFalse((self.store.root / "raw").exists())
        self.assertTrue(all(photo["artifact"].get("storage") == "external"
                            for photo in self.store.load()["photos"].values()))

    def test_unreviewed_photos_block_and_request_specific_views(self):
        ingest(self.store, self.photos, self.policy)
        report = validate(self.store, self.policy)
        self.assertFalse(report["ready"])
        self.assertEqual(report["accepted_count"], 0)
        self.assertEqual(set(report["missing_views"]), set(VIEWS))
        self.assertTrue(all("manual_review_required" in reasons
                            for reasons in report["rejected"].values()))
        self.assertEqual(report["requests"][0]["yaw"], 0)

    def test_coverage_requires_distinct_evidence(self):
        self.ready_photos()
        review_photo(self.store, self.ids[1], 0, 0, True, True, True)
        report = validate(self.store, self.policy)
        self.assertFalse(report["ready"])
        self.assertIn("left_30", report["missing_views"])

    def test_non_neutral_or_obstructed_photos_do_not_count(self):
        self.ready_photos()
        review_photo(self.store, self.ids[0], 0, 0, False, False, False)
        report = validate(self.store, self.policy)
        self.assertEqual(set(report["rejected"][self.ids[0]]),
                         {"non_neutral_expression", "face_obstructed",
                          "inconsistent_lighting_or_focal_length"})

    def test_landmark_and_angle_validation(self):
        self.ready_photos()
        for yaw, pitch, landmarks in (
            (float("nan"), 0, None), (0, 91, None),
            (0, 0, {"eye": [float("nan"), 0.5]}),
            (0, 0, {"eye": [0.2]}), (0, 0, {}),
        ):
            with self.subTest(yaw=yaw, pitch=pitch, landmarks=landmarks):
                with self.assertRaises(ValueError):
                    review_photo(self.store, self.ids[0], yaw, pitch,
                                 True, True, True, landmarks)
        result = review_photo(self.store, self.ids[0], 0, 0, True, True, True,
                              {"nose": [0.5, 0.6]})
        self.assertEqual(result["landmarks"]["nose"], [0.5, 0.6])

    def test_stage_order_and_exact_handoff_contract(self):
        with self.assertRaisesRegex(ValueError, "Complete intake"):
            attach(self.store, self.policy, "cleanup",
                   {"mesh": self.mesh, "texture": self.texture}, "done")
        self.ready_photos()
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            attach(self.store, self.policy, "reconstruction", {"mesh": self.mesh}, "done")

    def test_complete_manual_workflow_survives_reload(self):
        self.metahuman()
        pairs = {"front": (self.ids[0], self.texture),
                 "three_quarter": (self.ids[3], self.texture),
                 "profile": (self.ids[5], self.texture)}
        result = baseline(self.store, self.policy, pairs, "Reviewed synthetic comparison")
        self.assertEqual(result["mode"], "human_visual_baseline")
        with Image.open(self.store.resolve(result["artifacts"]["comparison"]["path"])) as sheet:
            self.assertEqual(sheet.size, (960, 1200))
        reloaded = Store(self.root / "data", "subject-01")
        self.assertTrue(status(reloaded)["baseline_recorded"])
        self.assertFalse(status(reloaded)["production_ready"])

    def test_changed_review_invalidates_downstream_and_keeps_history(self):
        self.metahuman()
        old = self.store.load()["stages"]["reconstruction"]
        review_photo(self.store, self.ids[0], 1, 0, True, True, True)
        state = self.store.load()
        self.assertTrue(all(stage["status"] == "pending" for stage in state["stages"].values()))
        self.assertTrue(any(entry["previous"] == old for entry in state["history"]))
        self.assertTrue(all(self.store.verify(artifact)
                            for artifact in old["artifacts"].values()))

    def test_repeated_intake_and_validation_preserve_completed_work(self):
        self.metahuman()
        before = self.store.load()
        ingest(self.store, self.photos, self.policy)
        self.assertTrue(validate(self.store, self.policy)["ready"])
        self.assertEqual(self.store.load(), before)

    def test_changed_policy_requires_revalidation(self):
        self.ready_photos()
        self.reconstruction()
        new_policy = replace(self.policy, min_photos=9)
        with self.assertRaisesRegex(ValueError, "policy changed"):
            attach(self.store, new_policy, "cleanup",
                   {"mesh": self.mesh, "texture": self.texture}, "done")

    def test_camera_contract_requires_all_accepted_photos_and_finite_matrices(self):
        self.ready_photos()
        self.reconstruction()
        cameras = json.loads(self.cameras.read_text())
        cameras[self.ids[0]]["intrinsics"][0][0] = float("nan")
        self.cameras.write_text(json.dumps(cameras))
        artifacts = {"mesh": self.mesh, "texture": self.texture, "cameras": self.cameras}
        with self.assertRaisesRegex(ValueError, "finite"):
            attach(self.store, self.policy, "reconstruction", artifacts, "invalid")
        cameras.pop(self.ids[0])
        self.cameras.write_text(json.dumps(cameras))
        with self.assertRaisesRegex(ValueError, "every accepted"):
            attach(self.store, self.policy, "reconstruction", artifacts, "invalid")
        self.assertEqual(self.store.load()["stages"]["reconstruction"]["status"], "complete")

    def test_tampered_photo_blocks_downstream(self):
        self.ready_photos()
        artifact = self.store.load()["photos"][self.ids[0]]["artifact"]
        self.store.artifact_path(artifact).write_bytes(b"changed")
        self.assertTrue(status(self.store)["integrity_problems"])
        with self.assertRaisesRegex(ValueError, "Photo evidence changed"):
            self.reconstruction()
        self.assertFalse(validate(self.store, self.policy)["ready"])

    def test_missing_external_source_is_not_silently_replaced_by_a_copy(self):
        self.ready_photos()
        artifact = self.store.load()["photos"][self.ids[0]]["artifact"]
        self.store.artifact_path(artifact).unlink()
        report = validate(self.store, self.policy)
        self.assertIn("source_missing_or_changed", report["rejected"][self.ids[0]])

    def test_tampered_handoff_blocks_next_stage(self):
        self.ready_photos()
        reconstruction = self.reconstruction()
        self.store.resolve(reconstruction["artifacts"]["mesh"]["path"]).write_bytes(b"tamper")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            attach(self.store, self.policy, "cleanup",
                   {"mesh": self.mesh, "texture": self.texture}, "done")

    def test_baseline_requires_accepted_references(self):
        self.metahuman()
        with self.assertRaisesRegex(ValueError, "accepted photo"):
            baseline(self.store, self.policy, {
                "front": ("unknown", self.texture),
                "three_quarter": (self.ids[3], self.texture),
                "profile": (self.ids[5], self.texture),
            }, "review")

    def test_baseline_rejects_reference_with_wrong_view(self):
        self.metahuman()
        with self.assertRaisesRegex(ValueError, "camera angle"):
            baseline(self.store, self.policy, {
                "front": (self.ids[5], self.texture),
                "three_quarter": (self.ids[3], self.texture),
                "profile": (self.ids[5], self.texture),
            }, "review")

    def test_baseline_comparison_history_is_preserved(self):
        self.metahuman()
        pairs = {view: (self.ids[index], self.texture)
                 for view, index in (("front", 0), ("three_quarter", 3), ("profile", 5))}
        old = baseline(self.store, self.policy, pairs, "first")
        self.image(self.texture, 150)
        baseline(self.store, self.policy, pairs, "second")
        state = self.store.load()
        self.assertNotEqual(state["stages"]["baseline"]["artifacts"]["comparison"],
                            old["artifacts"]["comparison"])
        self.assertTrue(self.store.verify(old["artifacts"]["comparison"]))
        self.assertTrue(any(entry["previous"] == old for entry in state["history"]))

    def test_empty_or_wrong_artifacts_do_not_complete_handoff(self):
        self.ready_photos()
        self.reconstruction()
        self.mesh.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            attach(self.store, self.policy, "cleanup",
                   {"mesh": self.mesh, "texture": self.texture}, "done")
        self.assertEqual(self.store.load()["stages"]["cleanup"]["status"], "pending")

    def test_lock_prevents_competing_writer_and_cleans_up(self):
        with self.store.lock():
            with self.assertRaisesRegex(ValueError, "locked"):
                with self.store.lock():
                    self.fail("Second writer obtained lock")
        with self.store.lock():
            self.assertTrue((self.store.root / ".lock").exists())
        self.assertFalse((self.store.root / ".lock").exists())

    def test_paths_cannot_escape_and_existing_session_is_not_overwritten(self):
        for name in ("..", r"..\other", "x/y", "x:y"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Store(self.root, name)
        with self.assertRaises(ValueError):
            self.store.resolve(r"..\outside.png")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.initialize()

    def test_cli_json_exit_codes_and_persistence(self):
        config = self.root / "settings.json"
        config.write_text(json.dumps({"data_root": "cli-data"}))
        options = ["--config", str(config)]
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            self.assertEqual(main([*options, "init", "cli"]), 0)
            self.assertEqual(main([*options, "validate", "cli"]), 2)
            self.assertEqual(main([*options, "init", "cli"]), 1)
        self.assertIn('"ready": false', output.getvalue())
        self.assertEqual(json.loads(error.getvalue())["error"], "ValueError")
        self.assertEqual(Store(self.root / "cli-data", "cli").load()["schema_version"], 1)

    def test_configuration_rejects_invalid_values_and_resolves_relative_paths(self):
        path = self.root / "settings.json"
        for value in ({"min_photos": 20}, {"min_photos": True}, {"angle_tolerance": 20},
                      {"max_dark_fraction": -1}, {"unknown": 2}, {"min_short_side": 8.5}):
            path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                load_settings(path)
        path.write_text(json.dumps({"data_root": "local-data"}))
        _, root = load_settings(path)
        self.assertEqual(root, (self.root / "local-data").resolve())


if __name__ == "__main__":
    unittest.main()
