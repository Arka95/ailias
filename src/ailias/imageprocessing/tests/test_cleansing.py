import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from ailias.generation.inputs import prepare_inputs
from ailias.generation.settings import GenerationPolicy
from ailias.imageprocessing.backend import Face
from ailias.imageprocessing.cleanup import cleanup_alias
from ailias.imageprocessing.cleansing import (
    DEFAULT_CONFIG, CleansingPolicy, appearance, cleanse_run, foreground, load_cleansing, white_balance,
)
from ailias.imageprocessing.curation import PROFILE, capture_curation
from ailias.imageprocessing.pipeline import load_run, scan
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.storage import Store, atomic_json, digest


class FakeDetector:
    provenance = {"provider": "cleansing-fixture", "revision": "1", "models": {"sface": "fixture"}}

    def detect(self, image):
        return [Face([40, 40, 80, 80], [[55, 60], [100, 65], [80, 80], [60, 105], [100, 110]], 0.99, [1., 0.])]


class FakeMatting:
    def __init__(self, fail_at=None, opaque=False):
        self.calls, self.fail_at, self.opaque = 0, fail_at, opaque
        self.provenance = {"provider": "fake-matting", "device": "unit-test"}
        self.closed = False

    def matte(self, image):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError("fixture inference failure")
        alpha = Image.new("L", image.size, 255 if self.opaque else 0)
        if not self.opaque:
            draw = ImageDraw.Draw(alpha)
            draw.rectangle((30, 20, image.width - 30, image.height - 1), fill=128)
            draw.rectangle((35, 25, image.width - 35, image.height - 1), fill=255)
        return alpha

    def close(self):
        self.closed = True

    def parse(self, image, bounds):
        parts = Image.new("L", image.size, 1 if self.opaque else 0)
        if not self.opaque:
            draw = ImageDraw.Draw(parts)
            draw.rectangle((30, 20, image.width - 30, int(image.height*0.78)), fill=1)
            draw.rectangle((30, 12, image.width - 30, 25), fill=17)
            draw.rectangle((0, int(image.height*0.79), image.width - 1, image.height - 1), fill=16)
        return parts


class AppearanceTests(unittest.TestCase):
    def test_head_extraction_removes_shirt_neck_and_other_heads_without_losing_hair_or_ears(self):
        from ailias.imageprocessing.faceparsing import head_alpha
        alpha = Image.new("L", (100, 120), 255)
        parts = Image.new("L", alpha.size, 0)
        draw = ImageDraw.Draw(parts)
        draw.rectangle((25, 30, 75, 75), fill=1)
        draw.rectangle((25, 15, 75, 35), fill=17)
        draw.rectangle((20, 40, 25, 55), fill=7)
        draw.rectangle((45, 76, 55, 85), fill=14)
        draw.rectangle((10, 86, 90, 119), fill=16)
        draw.rectangle((0, 0, 10, 10), fill=1)
        draw.rectangle((35, 0, 65, 14), fill=18)
        draw.point((19, 45), fill=9)
        alpha.putpixel((24, 22), 80)
        result, record = head_alpha(alpha, parts, [25, 30, 50, 45])
        self.assertEqual(result.getpixel((50, 100)), 0, "Clothing is not face foreground")
        self.assertEqual(result.getpixel((50, 80)), 0, "Neck is excluded from head-only output")
        self.assertEqual(result.getpixel((5, 5)), 0, "An unrelated head is not the selected subject")
        self.assertEqual(result.getpixel((50, 20)), 255)
        self.assertEqual(result.getpixel((22, 45)), 255)
        self.assertEqual(result.getpixel((24, 22)), 80, "Preserve the soft external hair fringe")
        self.assertEqual(result.getpixel((50, 14)), 0, "Do not grow the hair fringe into a hat")
        self.assertEqual(result.getpixel((19, 45)), 0, "Do not grow ears into jewellery")
        self.assertEqual(record["target"], "head")

    def test_head_extraction_requires_valid_native_coordinate_labels(self):
        from ailias.imageprocessing.faceparsing import head_alpha
        alpha = Image.new("L", (100, 100), 255)
        for parts in (None, Image.new("RGB", alpha.size), Image.new("L", (50, 50)),
                      Image.new("L", alpha.size, 255)):
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                head_alpha(alpha, parts, [25, 25, 50, 50])

    def test_head_semantics_reaching_into_lower_context_are_flagged_for_review(self):
        from ailias.imageprocessing.faceparsing import head_alpha
        parts = Image.new("L", (100, 160))
        ImageDraw.Draw(parts).rectangle((25, 20, 75, 159), fill=1)
        alpha, record = head_alpha(Image.new("L", parts.size, 255), parts, [25, 30, 50, 60])
        self.assertEqual(alpha.getpixel((50, 150)), 0)
        self.assertEqual(record["review_reason"], "semantic_head_reaches_lower_context_boundary")

    def test_white_balance_uses_scene_neutrals_without_neutralizing_skin(self):
        scene = Image.new("RGB", (100, 100), (150, 145, 140))
        ImageDraw.Draw(scene).rectangle((30, 20, 70, 80), fill=(165, 100, 70))
        policy = CleansingPolicy()
        balance = white_balance(scene, [[30, 20, 40, 60]], policy)
        self.assertEqual(balance["status"], "estimated_scene_neutrals")
        corrected, record = appearance(scene, [30, 20, 40, 60], balance, policy)
        neutral, skin = np.asarray(corrected)[0, 0], np.asarray(corrected)[40, 40]
        self.assertLessEqual(int(neutral.max()) - int(neutral.min()), 1)
        self.assertGreater(int(skin[0]), int(skin[2]) + 50)
        self.assertFalse(record["intrinsic_albedo"])

    def test_skin_only_scene_does_not_invent_a_neutral_reference(self):
        balance = white_balance(Image.new("RGB", (100, 100), (160, 95, 55)), [], CleansingPolicy())
        self.assertTrue(balance["status"].startswith("skipped_"))
        self.assertEqual(balance["gains"], [1., 1., 1.])

    def test_relative_exposure_is_bounded_and_highlights_are_not_forced_to_clip(self):
        policy = CleansingPolicy()
        balance = {"gains": [1., 1., 1.], "status": "fixture"}
        image = Image.new("RGB", (50, 50), (100, 80, 60))
        brighter, record = appearance(image, [0, 0, 50, 50], balance, policy, target=0.8)
        self.assertAlmostEqual(record["exposure_ev"], policy.exposure_ev_limit)
        self.assertGreater(brighter.getpixel((25, 25))[0], 100)
        ImageDraw.Draw(image).rectangle((0, 0, 20, 49), fill=(250, 250, 250))
        _, record = appearance(image, [20, 0, 30, 50], balance, policy, target=0.8)
        self.assertEqual(record["exposure_ev"], 0)
        self.assertEqual(record["exposure_status"], "skipped_highlight_clipping")

    def test_foreground_preserves_soft_alpha_but_masks_native_rgb_background(self):
        image = Image.new("RGB", (100, 100), (160, 100, 60))
        alpha = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(alpha)
        draw.rectangle((20, 20, 80, 99), fill=255)
        draw.line((19, 20, 19, 99), fill=80)
        products, record = foreground(image, alpha, [25, 25, 50, 50], CleansingPolicy())
        rgb, cutout, matte = products
        self.assertEqual(record["status"], "ready")
        self.assertEqual(rgb.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(cutout.getpixel((0, 0)), (0, 0, 0, 0))
        self.assertEqual(cutout.getpixel((19, 50)), (160, 100, 60, 80))
        self.assertEqual(rgb.getpixel((19, 50)), (0, 0, 0))
        self.assertEqual(matte.size, image.size)

    def test_empty_opaque_or_misaligned_alpha_is_not_claimed_as_removed_background(self):
        image = Image.new("RGB", (100, 100))
        for value in (0, 255):
            products, record = foreground(image, Image.new("L", image.size, value),
                                           [25, 25, 50, 50], CleansingPolicy())
            self.assertIsNone(products)
            self.assertEqual(record["status"], "review")
        with self.assertRaisesRegex(ValueError, "native-size"):
            foreground(image, Image.new("L", (50, 50)), [25, 25, 50, 50], CleansingPolicy())


class CleansingPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / "data", "subject")
        self.input = self.root / "photos"
        self.input.mkdir()
        for index in range(3):
            image = Image.new("RGB", (160, 160), (150, 145, 140))
            ImageDraw.Draw(image).ellipse((35, 20, 125, 150), fill=(150 + index*10, 100, 70))
            image.save(self.input / f"{index}.png")
        self.processing = ProcessingPolicy(min_face_size=8, min_laplacian_variance=0)
        self.scan = scan(self.store, self.input, self.processing, FakeDetector(), body=False)
        self.run = self.scan["run"]
        capture_curation(self.store, self.run, self.processing, removed_are_rejected=True)
        self.policy = CleansingPolicy()
        self.generation = GenerationPolicy(max_images=3, min_registered_images=3,
                                           use_cleansed_portraits=True)

    def cleanse(self, backend=None):
        return cleanse_run(self.store, self.run, self.policy, backend or FakeMatting())

    def test_publishes_derivatives_without_changing_reviewed_portraits_sources_or_curation(self):
        before = {path: digest(path) for path in self.input.glob("*.png")}
        report = load_run(self.store, self.run)
        portraits = [self.store.resolve(face["products"]["portrait"]) for face in report["faces"].values()]
        before.update({path: digest(path) for path in portraits})
        before[self.store.resolve(PROFILE)] = digest(self.store.resolve(PROFILE))
        backend = FakeMatting()
        result = self.cleanse(backend)
        self.assertEqual(result["summary"]["ready"], 3)
        self.assertEqual(backend.calls, 3)
        self.assertEqual(before, {path: digest(path) for path in before})
        for record in result["faces"].values():
            for name, mode in (("rgb", "RGB"), ("cutout", "RGBA"), ("alpha", "L")):
                self.assertTrue(self.store.verify(record["products"][name]))
                with Image.open(self.store.artifact_path(record["products"][name])) as image:
                    self.assertEqual(image.mode, mode)
                    self.assertEqual(list(image.size), record["size"])
            with Image.open(self.store.artifact_path(record["products"]["aligned"])) as aligned:
                self.assertEqual(aligned.size, (512, 512))
                self.assertEqual(aligned.mode, "RGBA")
            self.assertTrue(record["alignment"]["analysis_only"])
            self.assertFalse(record["alignment"]["yaw_pitch_frontalized"])
        shown = [key for page in result["preview_pages"] for key in page["face_ids"]]
        self.assertEqual(set(shown), set(result["faces"]))
        self.assertEqual(len(shown), len(set(shown)))
        self.assertEqual(result["cleanup"]["errors"], 0)

    def test_generation_uses_corrected_rgb_and_alpha_masks_and_reuses_input_hardlinks(self):
        self.cleanse()
        first = prepare_inputs(self.store, self.run, "face-001", self.generation)
        second = prepare_inputs(self.store, self.run, "face-001", self.generation)
        self.assertEqual(first["key"], second["key"])
        self.assertEqual(first["inputs"]["mask"], "portrait-alpha-threshold-v1")
        root = self.store.resolve(first["manifest"]).parent
        entry = first["inputs"]["entries"][0]
        self.assertNotEqual(entry["image"]["path"], entry["portrait"]["path"])
        with Image.open(root / "images" / entry["image_name"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(image.getpixel((80, 140)), (0, 0, 0), "Shirt pixels never reach native tools")
        with Image.open(root / "masks" / (entry["image_name"] + ".png")) as mask:
            self.assertEqual(mask.getpixel((0, 0)), 0)
            self.assertEqual(mask.getpixel((80, 80)), 255)
            self.assertEqual(mask.getpixel((80, 140)), 0, "Person foreground is not head foreground")
        self.assertFalse(entry["cleansing"]["color"]["intrinsic_albedo"])
        self.assertNotEqual(entry["analysis"]["aligned_cutout"]["path"], entry["image"]["path"])
        self.assertTrue(entry["analysis"]["native_projection_preserved"])

    def test_preview_paginates_every_ready_head_without_dropping_faces_after_36(self):
        report = load_run(self.store, self.run)
        template = next(iter(report["faces"].values()))
        report["faces"] = {f"observation-{index}": dict(template, id=f"observation-{index}")
                           for index in range(37)}
        report["groups"] = [{"id": "face-001", "members": list(report["faces"])}]
        with patch("ailias.imageprocessing.cleansing.load_run", return_value=report):
            result = self.cleanse()
        shown = [key for page in result["preview_pages"] for key in page["face_ids"]]
        self.assertEqual(len(result["preview_pages"]), 2)
        self.assertEqual(set(shown), set(report["faces"]))
        self.assertEqual(len(shown), 37)
        self.assertTrue(all(self.store.verify(page["artifact"]) for page in result["preview_pages"]))

    def test_native_stages_receive_actual_alpha_mask_directory(self):
        from ailias.generation.pipeline import reconstruct
        from ailias.generation.tests.test_generation import FakeTools
        self.cleanse()
        tools = FakeTools()
        policy = replace(self.generation, min_sparse_points=1, min_fused_points=1, min_mesh_faces=1)
        with patch.object(tools, "run", wraps=tools.run) as native:
            result = reconstruct(self.store, self.run, "face-001", tools, policy)
        self.assertEqual(result["status"], "complete")
        options = {call.args[1]: dict(zip(call.args[2][1::2], call.args[2][2::2]))
                   for call in native.call_args_list if call.args[0] == "colmap"}
        masks = Path(options["feature-extraction"]["--ImageReader.mask_path"])
        self.assertEqual(str(masks), options["surface-fusion"]["--StereoFusion.mask_path"])
        self.assertTrue(list(masks.glob("*.png.png")))

    def test_generation_cli_refreshes_cleansing_and_closes_backend_before_native_reconstruction(self):
        from ailias.generation.cli import execute
        config = self.root / "generation.json"
        atomic_json(config, {"tools_root": str(self.root / "tools"), "policy": asdict(self.generation)})
        args = SimpleNamespace(alias="subject", generation_action="reconstruct",
                               generation_config=config, run=self.run, group="face-001")
        matting = FakeMatting()

        def reconstructed(*args):
            self.assertTrue(matting.closed)
            return {"status": "complete"}

        with patch("ailias.imageprocessing.matting.MattingBackend", return_value=matting), \
                patch("ailias.imageprocessing.cleansing.cleanse_run") as cleansing, \
                patch("ailias.generation.pipeline.reconstruct", side_effect=reconstructed) as native:
            result, code = execute(args, self.store.root.parent)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "complete")
        cleansing.assert_called_once()
        self.assertTrue(matting.closed)
        native.assert_called_once()

    def test_generation_refuses_missing_stale_or_changed_derivatives(self):
        with self.assertRaisesRegex(ValueError, "cleansing is required"):
            prepare_inputs(self.store, self.run, "face-001", self.generation)
        result = self.cleanse()
        record = next(iter(result["faces"].values()))
        path = self.store.artifact_path(record["products"]["alpha"])
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "alpha changed"):
            prepare_inputs(self.store, self.run, "face-001", self.generation)
        path.write_bytes(original)
        profile = json.loads(self.store.resolve(PROFILE).read_text())
        profile["note"] = "new review"
        atomic_json(self.store.resolve(PROFILE), profile)
        with self.assertRaisesRegex(ValueError, "stale"):
            load_cleansing(self.store, self.run, digest(self.store.resolve(PROFILE)))

    def test_bad_matte_stays_review_and_does_not_silently_revert_to_ellipse(self):
        result = self.cleanse(FakeMatting(opaque=True))
        self.assertEqual(result["summary"]["review"], 3)
        self.assertTrue(all(not record["products"] for record in result["faces"].values()))
        with self.assertRaisesRegex(ValueError, "Seed group has no"):
            prepare_inputs(self.store, self.run, "face-001", self.generation)

    def test_absent_head_semantics_are_review_not_a_fallback_to_person_cutouts(self):
        backend = FakeMatting()
        with patch.object(backend, "parse", side_effect=lambda image, bounds: Image.new("L", image.size)):
            result = self.cleanse(backend)
        self.assertEqual(result["summary"]["review"], 3)
        self.assertTrue(all(not record["products"] for record in result["faces"].values()))

    def test_face_coverage_alone_cannot_approve_a_semantic_mask_leaking_into_torso(self):
        backend = FakeMatting()
        def leaking_parts(image, bounds):
            parts = Image.new("L", image.size)
            ImageDraw.Draw(parts).rectangle((35, 20, image.width - 35, image.height - 1), fill=1)
            return parts
        with patch.object(backend, "parse", side_effect=leaking_parts):
            result = self.cleanse(backend)
        self.assertEqual(result["summary"]["review"], 3)
        for record in result["faces"].values():
            self.assertEqual(record["quality"]["face_core_coverage"], 1.)
            self.assertEqual(record["quality"]["reason"], "semantic_head_reaches_lower_context_boundary")
            self.assertFalse(record["products"])

    def test_failed_fresh_inference_keeps_previous_outputs_and_cleans_partial_work(self):
        previous = self.cleanse()
        pointer = self.store.resolve(str(Path("imageprocessing") / "cleansing" / f"{self.run}.json"))
        before = pointer.read_bytes()
        with self.assertRaisesRegex(ValueError, "fixture inference"):
            self.cleanse(FakeMatting(fail_at=2))
        self.assertEqual(pointer.read_bytes(), before)
        self.assertTrue(all(self.store.verify(artifact) for record in previous["faces"].values()
                            for artifact in record["products"].values()))
        reports = list(pointer.parent.glob("*\\manifest.json"))
        failed = [json.loads(path.read_text()) for path in reports]
        self.assertTrue(any(record["status"] == "failed" for record in failed))
        self.assertFalse(list(pointer.parent.rglob("*.tmp")))

    def test_explicit_cleanup_protects_cleansing_products(self):
        result = self.cleanse()
        cleanup_alias(self.store)
        self.assertTrue(all(self.store.verify(artifact) for record in result["faces"].values()
                            for artifact in record["products"].values()))

    def test_cli_scan_runs_cleansing_without_overwriting_canonical_portraits(self):
        from ailias.imageprocessing.cli import execute
        config = self.root / "processing.json"
        atomic_json(config, asdict(self.processing))
        args = SimpleNamespace(alias="subject", image_action="scan", source=self.input,
                               processing_config=config, device="cpu", no_body=True,
                               recursive=False, limit=None, no_cleanse=False, cleansing_config=DEFAULT_CONFIG)
        matting = FakeMatting()
        with patch("ailias.imageprocessing.backend.OpenCVBackend", return_value=FakeDetector()), \
                patch("ailias.imageprocessing.matting.MattingBackend", return_value=matting):
            result, code = execute(args, self.store.root.parent)
        self.assertEqual(code, 0)
        self.assertEqual(result["cleansing"]["summary"]["ready"], 3)
        self.assertTrue(matting.closed)

    def test_repeated_cleansing_performs_fresh_inference_but_reuses_identical_assets(self):
        first = self.cleanse()
        backend = FakeMatting()
        second = self.cleanse(backend)
        self.assertEqual(backend.calls, 3)
        self.assertNotEqual(first["job"], second["job"])
        self.assertEqual([record["products"] for record in first["faces"].values()],
                         [record["products"] for record in second["faces"].values()])


if __name__ == "__main__":
    unittest.main()
