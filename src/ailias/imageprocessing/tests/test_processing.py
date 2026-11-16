import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.imageprocessing.backend import Face
from ailias.imageprocessing.features import group_faces, portrait_box, subject_bodies
from ailias.imageprocessing.models import download, verified_models
from ailias.imageprocessing.pipeline import load_run, read_image, scan, select_group
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.storage import Store
from ailias.resources import cleanup_owned_files


def face(x=100, embedding=None):
    return Face(
        box=[x, 130, 120, 160],
        landmarks=[[x + 30, 180], [x + 90, 180], [x + 60, 210],
                   [x + 40, 240], [x + 80, 240]],
        score=0.99, embedding=embedding or [1.0, 0.0],
    )


class FakeBackend:
    provenance = {"provider": "fake-unit-backend", "device": "cpu"}

    def __init__(self, faces=None):
        self.faces = faces if faces is not None else [face()]
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.faces

    def segment(self, image):
        values = np.zeros((image.height, image.width), dtype=np.uint8)
        values[80:image.height - 10, 30:image.width - 30] = 255
        return Image.fromarray(values)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / "data", "arka")
        self.input = self.root / "input"
        self.input.mkdir()
        self.policy = ProcessingPolicy(min_face_size=80, min_laplacian_variance=1)
        y, x = np.indices((500, 400))
        values = ((x * 41 + y * 31) % 180 + 35).astype(np.uint8)
        self.image = Image.fromarray(values).convert("RGB")
        self.source = self.input / "photo.png"
        self.image.save(self.source)

    def run_scan(self, backend=None, **options):
        return scan(self.store, self.input, self.policy, backend or FakeBackend(), **options)

    def test_scan_exports_original_pixels_portrait_and_visible_body(self):
        original = self.source.read_bytes()
        result = self.run_scan()
        report = load_run(self.store, result["run"])
        self.assertEqual(result["summary"]["faces"], 1)
        self.assertEqual(result["summary"]["eligible_faces"], 1)
        self.assertEqual(result["summary"]["body_status"], {"visible_subject_only": 1})
        record = next(iter(report["faces"].values()))
        bounds = record["portrait_bounds"]
        with Image.open(self.store.resolve(record["products"]["portrait"])) as portrait:
            self.assertEqual(portrait.size, (bounds[2] - bounds[0], bounds[3] - bounds[1]))
            self.assertTrue(np.array_equal(np.asarray(portrait), np.asarray(self.image.crop(bounds))))
            self.assertNotIn("exif", portrait.info)
        with Image.open(self.store.resolve(record["products"]["body"])) as body:
            self.assertEqual(body.mode, "RGBA")
        self.assertEqual(self.source.read_bytes(), original)
        self.assertTrue(Path(result["overview"]).is_file())
        self.assertTrue(self.store.resolve(report["groups"][0]["contact_sheets"][0]).is_file())

    def test_detected_face_comparison_separates_identity_from_projected_geometry(self):
        from ailias.geometry.comparison import compare_detected_faces
        result = self.run_scan(body=False)
        report = load_run(self.store, result["run"])
        face_id = next(iter(report["faces"]))
        comparison = compare_detected_faces(self.store, result["run"], face_id, face_id)
        self.assertAlmostEqual(comparison["identity_cosine"], 1)
        self.assertLess(comparison["geometry_2d"]["normalized_rmse"], 1e-10)
        self.assertFalse(comparison["geometry_2d"]["is_3d_geometry"])
        self.assertTrue(Path(comparison["report"]).is_file())

    def test_exact_duplicates_are_analyzed_once(self):
        (self.input / "copy.png").write_bytes(self.source.read_bytes())
        backend = FakeBackend()
        result = self.run_scan(backend, body=False)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(result["summary"]["duplicates"], 1)
        self.assertEqual(result["summary"]["input_files"], 2)

    def test_corruption_and_no_face_are_reported_without_losing_valid_work(self):
        (self.input / "corrupt.jpg").write_bytes(b"bad")
        result = self.run_scan(FakeBackend([]))
        self.assertEqual(result["summary"]["no_face"], 1)
        self.assertEqual(result["summary"]["errors"], 1)
        self.assertEqual(result["summary"]["groups"], 0)
        self.assertTrue(Path(result["manifest"]).exists())
        self.assertEqual(result["cleanup"]["removed_files"], 0)
        report = load_run(self.store, result["run"])
        for source in report["sources"].values():
            self.assertTrue(self.store.verify(source["artifact"]))

    def test_failed_job_cleans_its_unreferenced_copies_without_touching_other_jobs_or_models(self):
        from ailias.imageprocessing.cuda import CudaUnavailable
        model = self.store.resolve(str(Path("imageprocessing") / "models" / "keep.onnx"))
        model.parent.mkdir(parents=True)
        model.write_bytes(b"not owned by scan")
        previous = self.run_scan(body=False)
        backend = FakeBackend()
        backend.detect = lambda image: (_ for _ in ()).throw(CudaUnavailable("simulated failure"))
        with self.assertRaises(CudaUnavailable):
            self.run_scan(backend, body=False)
        self.assertTrue(Path(previous["overview"]).is_file())
        self.assertEqual(model.read_bytes(), b"not owned by scan")
        runs = self.store.resolve(str(Path("imageprocessing") / "runs"))
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in runs.glob("*\\manifest.json")]
        self.assertTrue(any(record["status"] == "failed" for record in manifests))

    def test_cleanup_refuses_alias_root_and_keeps_referenced_results(self):
        directory = Path("imageprocessing") / "runs" / "owned"
        root = self.store.resolve(str(directory))
        root.mkdir(parents=True)
        (root / "keep.png").write_bytes(b"retained")
        (root / "unused.tmp").write_bytes(b"temporary")
        other = self.store.resolve(str(Path("imageprocessing") / "runs" / "other" / "keep.png"))
        other.parent.mkdir(parents=True)
        other.write_bytes(b"other job")
        result = cleanup_owned_files(self.store, str(directory), [str(directory / "keep.png")])
        self.assertEqual(result["removed_files"], 1)
        self.assertTrue((root / "keep.png").is_file())
        self.assertTrue(other.is_file())
        for directory in (".", r"..\other", "imageprocessing",
                          r"imageprocessing\runs", r"imageprocessing\models"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                cleanup_owned_files(self.store, directory, [])

    def test_recursion_is_explicit_and_never_ingests_own_output(self):
        nested = self.input / "nested"
        nested.mkdir()
        self.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(nested / "other.png")
        self.assertEqual(self.run_scan(body=False)["summary"]["unique_sources"], 1)
        self.assertEqual(self.run_scan(recursive=True, body=False)["summary"]["unique_sources"], 2)
        contained = Store(self.input, "output")
        first = scan(contained, self.input, self.policy, FakeBackend(), recursive=True, body=False)
        second = scan(contained, self.input, self.policy, FakeBackend(), recursive=True, body=False)
        self.assertEqual(first["summary"]["unique_sources"], 2)
        self.assertEqual(second["summary"]["unique_sources"], 2)

    def test_small_face_is_retained_for_review_but_not_selectable(self):
        result = scan(self.store, self.input, ProcessingPolicy(), FakeBackend(), body=False)
        report = load_run(self.store, result["run"])
        record = next(iter(report["faces"].values()))
        self.assertIn("face_too_small", record["quality"]["reasons"])
        with self.assertRaisesRegex(ValueError, "no eligible"):
            select_group(self.store, result["run"], "face-001")

    def test_multiple_people_in_portrait_are_flagged(self):
        result = self.run_scan(FakeBackend([face(80), face(150, [0.0, 1.0])]), body=False)
        report = load_run(self.store, result["run"])
        self.assertTrue(all("portrait_contains_other_face" in item["quality"]["reasons"]
                            for item in report["faces"].values()))
        self.assertEqual(result["summary"]["groups"], 2)

    def test_touching_people_do_not_get_misattributed_body(self):
        mask = Image.new("L", (800, 500), 255)
        bodies = subject_bodies(mask, [face(100), face(500)])
        self.assertEqual([body["status"] for body in bodies],
                         ["ambiguous_touching_subjects", "ambiguous_touching_subjects"])

    def test_select_is_a_new_snapshot_and_does_not_change_manual_pipeline_state(self):
        self.store.initialize()
        before = self.store.load()
        result = self.run_scan()
        selected = select_group(self.store, result["run"], "face-001")
        again = select_group(self.store, result["run"], "face-001")
        self.assertNotEqual(selected["selection"], again["selection"])
        self.assertEqual(selected["selected_count"], 1)
        self.assertEqual(len(selected["portraits"]), 1)
        self.assertTrue(Path(selected["portraits"][0]).is_file())
        self.assertEqual(selected["originals"], [str(self.source.resolve())])
        selection_files = list(Path(selected["manifest"]).parent.rglob("*"))
        self.assertEqual([path.name for path in selection_files], ["selection.json"])
        self.assertEqual(self.store.load(), before)
        for artifact in selected["members"][0]["products"].values():
            self.assertTrue(self.store.verify(artifact))

    def test_selection_rejects_changed_original(self):
        result = self.run_scan(body=False)
        report = load_run(self.store, result["run"])
        source = next(iter(report["sources"].values()))["artifact"]
        self.store.artifact_path(source).write_bytes(b"tamper")
        with self.assertRaisesRegex(ValueError, "changed"):
            select_group(self.store, result["run"], "face-001")

    def test_fresh_scans_recompute_inference_but_reuse_generated_assets(self):
        backend = FakeBackend()
        first = self.run_scan(backend)
        first_report = load_run(self.store, first["run"])
        assets = self.store.resolve(str(Path("imageprocessing") / "assets"))
        files = {path: path.stat().st_mtime_ns for path in assets.rglob("*.png")}
        second = self.run_scan(backend)
        second_report = load_run(self.store, second["run"])
        self.assertEqual(backend.calls, 2)
        self.assertEqual(files, {path: path.stat().st_mtime_ns for path in assets.rglob("*.png")})
        self.assertEqual(next(iter(first_report["faces"].values()))["products"],
                         next(iter(second_report["faces"].values()))["products"])
        for result in (first, second):
            self.assertFalse((Path(result["manifest"]).parent / "originals").exists())
        original = next(iter(second_report["sources"].values()))["artifact"]
        self.assertEqual(self.store.artifact_path(original), self.source.resolve())

    def test_selection_intake_reuses_assets_and_checks_their_integrity(self):
        from ailias.intake import ingest
        from ailias.settings import Policy
        self.store.initialize()
        result = self.run_scan(body=False)
        selected = select_group(self.store, result["run"], "face-001")
        self.assertEqual(ingest(self.store, Path(selected["manifest"]), Policy())["total"], 1)
        artifact = next(iter(self.store.load()["photos"].values()))["artifact"]
        self.assertEqual(str(self.store.artifact_path(artifact)), selected["portraits"][0])
        self.assertFalse((self.store.root / "raw").exists())
        self.store.artifact_path(artifact).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            ingest(self.store, Path(selected["manifest"]), Policy())

    def test_cleanup_migrates_legacy_copies_and_preserves_selected_runs(self):
        from ailias.imageprocessing.cleanup import cleanup_alias
        from ailias.storage import atomic_json
        keep = self.run_scan()
        manifest = Path(keep["manifest"])
        report = load_run(self.store, keep["run"])
        relative = manifest.parent.relative_to(self.store.root)
        source = next(iter(report["sources"].values()))
        source["artifact"] = self.store.copy_artifact(self.source, str(relative / "originals"))
        old_source = self.store.artifact_path(source["artifact"])
        old_products = []
        for record in report["faces"].values():
            for role, value in list(record["products"].items()):
                canonical = self.store.resolve(value)
                copy = self.store.copy_artifact(canonical, str(relative / "groups" / role))
                record["products"][role] = copy["path"]
                old_products.append(self.store.artifact_path(copy))
                canonical.unlink()
        atomic_json(manifest, report)
        obsolete = self.run_scan(body=False)
        protected = self.run_scan(body=False)
        selected = select_group(self.store, protected["run"], "face-001")
        result = cleanup_alias(self.store, keep["run"], prune_superseded=True)
        self.assertEqual(result["pruned_runs"], [obsolete["run"]])
        self.assertEqual(result["migrated_sources"], 1)
        self.assertEqual(result["reused_generated_assets"], 3)
        self.assertFalse(old_source.exists())
        self.assertTrue(all(not path.exists() for path in old_products))
        self.assertTrue(Path(selected["manifest"]).is_file())
        self.assertTrue(Path(protected["manifest"]).is_file())
        self.assertTrue(Path(keep["overview"]).is_file())
        updated = load_run(self.store, keep["run"])
        self.assertTrue(self.store.verify(next(iter(updated["sources"].values()))["artifact"]))
        self.assertTrue(all(self.store.resolve(value).is_file()
                            for record in updated["faces"].values()
                            for value in record["products"].values()))
        self.assertEqual(cleanup_alias(self.store)["removed_files"], 0)

    def test_cleanup_retains_the_only_copy_if_external_original_is_missing(self):
        from ailias.imageprocessing.cleanup import cleanup_alias
        from ailias.storage import atomic_json
        result = self.run_scan(body=False)
        report = load_run(self.store, result["run"])
        manifest = Path(result["manifest"])
        relative = manifest.parent.relative_to(self.store.root) / "originals"
        source = next(iter(report["sources"].values()))
        source["artifact"] = self.store.copy_artifact(self.source, str(relative))
        atomic_json(manifest, report)
        self.source.unlink()
        cleaned = cleanup_alias(self.store)
        self.assertEqual(cleaned["retained_missing_sources"], 1)
        self.assertTrue(self.store.verify(source["artifact"]))
        with self.assertRaises(ValueError):
            cleanup_alias(self.store, prune_superseded=True)

    def test_cleanup_does_not_migrate_a_pruned_manifest_when_directory_removal_fails(self):
        from ailias.imageprocessing.cleanup import cleanup_alias
        obsolete = self.run_scan(body=False)
        keep = self.run_scan(body=False)
        directory = Path(obsolete["manifest"]).parent
        rmdir = Path.rmdir

        def held_directory(path):
            if path == directory:
                raise PermissionError("directory temporarily held by indexer")
            return rmdir(path)

        with patch.object(Path, "rmdir", held_directory):
            result = cleanup_alias(self.store, keep["run"], prune_superseded=True)
        self.assertIn(obsolete["run"], result["pruned_runs"])
        self.assertIn(obsolete["run"], result["retained_empty_runs"])
        self.assertFalse(Path(obsolete["manifest"]).exists())
        self.assertTrue(Path(keep["overview"]).is_file())

    def test_selection_failure_cleans_uncommitted_artifacts_not_run_outputs(self):
        result = self.run_scan(body=False)
        from ailias.storage import atomic_json as save

        def failing_selection(path, value):
            if path.name == "selection.json":
                raise OSError("simulated manifest failure")
            return save(path, value)

        with patch("ailias.imageprocessing.pipeline.atomic_json", side_effect=failing_selection):
            with self.assertRaises(OSError):
                select_group(self.store, result["run"], "face-001")
        selections = self.store.resolve(str(Path("imageprocessing") / "selections"))
        self.assertFalse(any(path.is_file() for path in selections.rglob("*")))
        self.assertTrue(Path(result["overview"]).is_file())

    def test_selection_rejects_superseded_grouping_without_deleting_old_outputs(self):
        result = self.run_scan(body=False)
        path = Path(result["manifest"])
        report = json.loads(path.read_text(encoding="utf-8"))
        report.pop("grouping_method")
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "superseded grouping"):
            select_group(self.store, result["run"], "face-001")
        self.assertTrue(Path(result["overview"]).is_file())

    def test_run_identifiers_and_scan_limits_are_validated(self):
        with self.assertRaises(ValueError):
            load_run(self.store, r"..\other")
        with self.assertRaises(ValueError):
            self.run_scan(limit=0)
        for value in (float("nan"), -1):
            with self.assertRaises(ValueError):
                replace(self.policy, identity_cosine_threshold=value)

    def test_exif_orientation_is_normalized_without_changing_original(self):
        source = self.root / "oriented.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (80, 40), "blue").save(source, exif=exif)
        before = source.read_bytes()
        image, metadata = read_image(source)
        self.assertEqual(image.size, (40, 80))
        self.assertEqual(metadata["exif_orientation"], 6)
        self.assertEqual(source.read_bytes(), before)

    def test_heic_decoding_uses_the_primary_full_resolution_image(self):
        from pillow_heif import register_heif_opener
        register_heif_opener(thumbnails=False)
        source = self.root / "sample.heic"
        Image.new("RGB", (64, 48), "green").save(source, format="HEIF")
        image, metadata = read_image(source)
        self.assertEqual(image.size, (64, 48))
        self.assertEqual(metadata["oriented_size"], [64, 48])

    def test_crop_bounds_preserve_native_resolution_and_clamp_at_edges(self):
        self.assertEqual(portrait_box([0, 0, 100, 100], (400, 500), self.policy),
                         (0, 0, 160, 200))

    def test_fresh_scan_does_not_replay_cached_inference(self):
        backend = FakeBackend()
        first = self.run_scan(backend, body=False)
        second = self.run_scan(backend, body=False)
        self.assertNotEqual(first["run"], second["run"])
        self.assertEqual(backend.calls, 2)

    def test_model_download_verifies_before_replacing_existing_file(self):
        path = self.root / "model.onnx"
        path.write_bytes(b"previous")
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                download("https://example.test/model", path, 100, hashlib.sha256(b"expected").hexdigest())
        self.assertEqual(path.read_bytes(), b"previous")
        self.assertFalse(list(self.root.glob("*.download")))

    def test_missing_models_never_trigger_implicit_downloads(self):
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "run imageprocessing models"):
                verified_models(self.store, True)
            network.assert_not_called()


class GroupingTests(unittest.TestCase):
    def record(self, name, angle, size, source=None):
        import math
        angle = math.radians(angle)
        return {"id": name, "source_id": source or name, "score": 0.99,
                "quality": {"native_short_side": size}, "eligible": True,
                "embedding": [math.cos(angle), math.sin(angle)]}

    def test_complete_link_prevents_identity_bridge_merges(self):
        records = [self.record("a", 0, 300), self.record("b", 30, 290),
                   self.record("c", 60, 280)]
        groups = group_faces(records, ProcessingPolicy(identity_cosine_threshold=0.8))
        self.assertEqual(groups[0]["members"], ["a", "b"])
        self.assertEqual(groups[1]["members"], ["c"])

    def test_two_faces_in_one_photo_cannot_share_a_group(self):
        records = [self.record("a", 0, 300, "same"), self.record("b", 0, 290, "same")]
        self.assertEqual(len(group_faces(records, ProcessingPolicy())), 2)

    def test_ambiguous_match_is_flagged_after_clustering_and_not_export_eligible(self):
        records = [self.record("a", -30, 300), self.record("b", 30, 290),
                   self.record("c", 0, 280)]
        groups = group_faces(records, ProcessingPolicy(identity_cosine_threshold=0.8))
        self.assertEqual(len(groups), 2)
        self.assertTrue(records[-1]["identity_ambiguous"])
        self.assertEqual(sum(group["eligible"] for group in groups), 2)

    def test_ambiguous_faces_cannot_cascade_into_singleton_alias_explosion(self):
        records = [self.record("a", -30, 400), self.record("b", 30, 390)]
        records.extend(self.record(f"mid-{index:02d}", 0, 300 - index) for index in range(20))
        groups = group_faces(records, ProcessingPolicy(identity_cosine_threshold=0.8))
        self.assertEqual(len(groups), 2)
        self.assertEqual(sum(len(group["members"]) for group in groups), len(records))

    def test_empty_and_single_face_sets_are_valid(self):
        self.assertEqual(group_faces([], ProcessingPolicy()), [])
        groups = group_faces([self.record("a", 0, 300)], ProcessingPolicy())
        self.assertEqual(groups[0]["eligible"], 1)
