import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from ailias.generation.cameras import read_model
from ailias.generation.inputs import prepare_inputs
from ailias.generation.matching import COLMAP_PAIR_BASE, matching_summary
from ailias.generation.mesh import textured_ply_to_obj
from ailias.generation.pipeline import inspect_job, reconstruct
from ailias.generation.settings import GenerationPolicy
from ailias.generation.tools import MAX_LOG_BYTES, NativeTools, ToolError
from ailias.imageprocessing.cleanup import cleanup_alias
from ailias.imageprocessing.curation import METHOD, PROFILE, embedding_space
from ailias.imageprocessing.settings import ProcessingPolicy
from ailias.storage import Store, atomic_json, digest

RUN = "20261116T000000Z-abcd1234"
TEXTURED_TRIANGLE = (
    "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
    "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
    "property list uchar float texcoord\nend_header\n"
    "-0.4 -0.4 1\n0.4 -0.4 1\n0 0.4 1\n3 0 1 2 6 0 0 1 0 0.5 1\n"
)


def create_database(path, image_count, connected=True):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
        connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
        connection.executemany("INSERT INTO images VALUES (?,?)",
                               [(i, f"{i}.png") for i in range(1, image_count + 1)])
        if connected:
            connection.executemany("INSERT INTO two_view_geometries VALUES (?,?)",
                                   [(COLMAP_PAIR_BASE*i + i + 1, 25) for i in range(1, image_count)])


class FakeTools:
    def __init__(self, fail_at=None, connected=True, missing_render=False):
        self.fail_at, self.connected = fail_at, connected
        self.missing_render = missing_render
        self.operations = []

    def probe(self, name):
        return {"ready": True, "version": "fixture"}

    def run(self, name, operation, args, directory):
        self.operations.append(operation)
        logs = directory / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{operation}.log").write_text(
            "Creating SIFT GPU feature extractor\nCreating SIFT GPU feature matcher\n", encoding="ascii")
        if self.fail_at == operation:
            (directory / "work" / "partial.tmp").write_bytes(b"unfinished native output")
            raise ToolError("native fixture failed")
        values = dict(zip(args[1::2], args[2::2])) if name == "colmap" else {}
        if operation == "feature-extraction":
            images = list(Path(values["--image_path"]).glob("*.png"))
            create_database(Path(values["--database_path"]), len(images), self.connected)
        elif operation == "sparse-mapping":
            model = Path(values["--output_path"]) / "0"
            model.mkdir()
            (model / "images.bin").write_bytes(b"fixture")
        elif operation.startswith("export-cameras"):
            output = Path(values["--output_path"])
            inputs = json.loads((directory / "manifest.json").read_text())
            view = directory.parents[2] / inputs["input_view"]["path"]
            entries = json.loads(view.read_text())["inputs"]["entries"]
            cameras, images = [], []
            for i, entry in enumerate(entries, 1):
                w, h = entry["size"]
                cameras.append(f"{i} SIMPLE_PINHOLE {w} {h} 100 {w/2} {h/2}\n")
                images.append(f"{i} 1 0 0 0 0 0 0 {i} {entry['image_name']}\n{w/2} {h/2} 1\n")
            (output / "cameras.txt").write_text("".join(cameras))
            (output / "images.txt").write_text("".join(images))
            (output / "points3D.txt").write_text("1 0 0 1 128 128 128 0 1 0\n")
        elif operation in {"surface-fusion", "surface-meshing"}:
            Path(values["--output_path"]).write_text(TEXTURED_TRIANGLE)
        elif operation == "texture-atlas":
            output = Path(values["--output_path"])
            output.mkdir()
            (output / "mesh.ply").write_text(TEXTURED_TRIANGLE)
            Image.new("RGB", (16, 16), "gray").save(output / "texture.png")
        elif operation == "blender-cleanup":
            request = json.loads(Path(args[-1]).read_text())
            output = Path(request["output"])
            (output / "head.obj").write_text("v 0 0 1\nv 1 0 1\nv 0 1 1\nf 1 2 3\n")
            (output / "head.mtl").write_text("newmtl fixture\n")
            (output / "head.blend").write_bytes(b"fake Blender output; never counted as integration")
            for label in ("reference", "orbit_45", "orbit_90"):
                if label != "orbit_90" or not self.missing_render:
                    Image.new("RGB", (16, 16), "gray").save(output / f"{label}.png")
            atomic_json(Path(request["receipt"]), {
                "status": "complete", "faces": 1, "metric_scale_verified": False})
        return {"operation": operation, "tool": name, "seconds": 0}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "data", "subject")
        self.policy = GenerationPolicy(max_images=3, min_registered_images=3,
                                       min_sparse_points=1, min_fused_points=1, min_mesh_faces=1)
        backend = {"provider": "fixture", "revision": "fixture", "models": {"sface": "fixture.onnx"}}
        processing = ProcessingPolicy(min_face_size=8)
        report = {"schema_version": 1, "status": "complete", "run": RUN, "backend": backend,
                  "policy": asdict(processing), "faces": {}, "sources": {}, "groups": []}
        examples = []
        for i in range(4):
            source = self.root / f"photo-{i}.png"
            Image.new("RGB", (32, 32), (40 + 20*i, 60, 100)).save(source)
            source_id = digest(source)
            with Image.open(source) as image:
                crop = self.root / f"crop-{i}.png"
                image.crop((2, 2, 30, 30)).save(crop)
            checksum = digest(crop)
            relative = str(Path("imageprocessing") / "assets" / "portraits" / f"{checksum}.png")
            destination = self.store.resolve(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            crop.replace(destination)
            face_id = f"{source_id}-00"
            quality = {"native_short_side": 28, "laplacian_variance": 100 + i, "reasons": []}
            report["sources"][source_id] = {"artifact": self.store.reference_artifact(source)}
            report["faces"][face_id] = {
                "id": face_id, "source_id": source_id, "quality": quality,
                "products": {"portrait": relative}, "portrait_bounds": [2, 2, 30, 30],
                "box_xywh": [8, 8, 16, 16], "embedding": [1, 0],
                "eligible": True, "identity_ambiguous": True,
            }
            examples.append({
                "portrait_sha256": checksum, "label": "accept",
                "source_id": source_id, "portrait_bounds": [2, 2, 30, 30],
                "embedding": [1, 0], "quality": quality,
                "artifact": {"path": relative, "sha256": checksum},
            })
        report["groups"] = [{"id": "face-001", "members": list(report["faces"])}]
        self.report = report
        self.run_path = self.store.resolve(str(Path("imageprocessing") / "runs" / RUN / "manifest.json"))
        atomic_json(self.run_path, report)
        atomic_json(self.store.resolve(PROFILE), {
            "schema_version": 1, "method": METHOD, "alias": "subject",
            "embedding_space": embedding_space(backend), "examples": examples})

    def test_native_input_view_reuses_hardlinks_masks_and_human_decisions(self):
        first = prepare_inputs(self.store, RUN, "face-001", self.policy)
        manifest = self.store.resolve(first["manifest"])
        before = {p: p.stat().st_mtime_ns for p in manifest.parent.rglob("*") if p.is_file()}
        second = prepare_inputs(self.store, RUN, "face-001", self.policy)
        self.assertEqual(first["key"], second["key"])
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
        entry = first["inputs"]["entries"][0]
        portrait = self.store.resolve(entry["portrait"]["path"])
        linked = manifest.parent / "images" / entry["image_name"]
        self.assertTrue(os.path.samefile(portrait, linked))
        with Image.open(manifest.parent / "masks" / (entry["image_name"] + ".png")) as mask:
            self.assertEqual(mask.getpixel((0, 0)), 0)
            self.assertEqual(mask.getpixel((14, 14)), 255)
        self.assertFalse(self.store.path.exists(), "Experimental generation must not invent capture attestations")

    def test_deleted_or_changed_reviewed_portrait_is_not_recreated(self):
        entry = next(iter(self.report["faces"].values()))
        portrait = self.store.resolve(entry["products"]["portrait"])
        portrait.unlink()
        with self.assertRaisesRegex(ValueError, "missing/changed"):
            prepare_inputs(self.store, RUN, "face-001", self.policy)
        self.assertFalse(portrait.exists())

    def test_quality_rejected_sources_do_not_enter_reconstruction(self):
        face = next(iter(self.report["faces"].values()))
        face["quality"]["reasons"] = ["face_too_small"]
        atomic_json(self.run_path, self.report)
        result = prepare_inputs(self.store, RUN, "face-001", self.policy)
        self.assertNotIn(face["source_id"], [e["source_id"] for e in result["inputs"]["entries"]])

    def test_cross_group_selection_uses_reviewed_support_and_keeps_view_provenance(self):
        ids = list(self.report["faces"])
        self.report["groups"] = [{"id": "face-001", "members": ids[:2]},
                                 {"id": "face-002", "members": ids[2:]}]
        atomic_json(self.run_path, self.report)
        result = prepare_inputs(self.store, RUN, "face-001", self.policy)
        self.assertIn("face-002", result["selection"]["groups_used"])
        self.assertEqual(result["selection"]["associated_count"], 4)
        self.assertTrue(all("view" in entry and "selection_weight" in entry
                            for entry in result["inputs"]["entries"]))
        self.assertEqual(result["selection"]["seed_group"], "face-001")
        seed_only = replace(self.policy, selection_expand_groups=False, min_registered_images=3)
        with self.assertRaisesRegex(ValueError, "Too few"):
            prepare_inputs(self.store, RUN, "face-001", seed_only)

    def test_explicit_warning_policy_widens_reviewed_inputs_without_hiding_quality(self):
        for face in self.report["faces"].values():
            face["quality"]["native_short_side"] = 128
        warned = next(iter(self.report["faces"].values()))
        warned["quality"]["reasons"] = ["face_underexposed"]
        atomic_json(self.run_path, self.report)
        result = prepare_inputs(self.store, RUN, "face-001",
                                replace(self.policy, max_images=4, selection_allow_quality_warnings=True))
        entry = next(entry for entry in result["inputs"]["entries"] if entry["face_id"] == warned["id"])
        self.assertEqual(entry["quality_warnings"], ["face_underexposed"])
        self.assertLess(entry["selection_weight"], max(e["selection_weight"]
                                                       for e in result["inputs"]["entries"]))

    def test_failed_input_publication_removes_its_partial_files(self):
        with patch("ailias.generation.inputs.os.link", side_effect=OSError("hardlinks unavailable")):
            with self.assertRaises(OSError):
                prepare_inputs(self.store, RUN, "face-001", self.policy)
        self.assertFalse(list(self.store.resolve("generation").rglob("sources.json")))
        self.assertFalse(list(self.store.resolve("generation").rglob("*.png")))

    def test_disconnected_view_graph_blocks_before_mapping_or_fabricating_mesh(self):
        tools = FakeTools(connected=False)
        result = reconstruct(self.store, RUN, "face-001", tools, self.policy)
        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("sparse-mapping", tools.operations)
        self.assertFalse(result["artifacts"])
        self.assertFalse((Path(result["manifest"]).parent / "work").exists())
        self.assertFalse(self.store.path.exists())

    def test_successful_native_outputs_commit_cameras_and_cleanup_and_remove_scratch(self):
        result = reconstruct(self.store, RUN, "face-001", FakeTools(), self.policy)
        self.assertEqual(result["status"], "complete", result)
        self.assertIn("cleaned_mesh", result["artifacts"])
        self.assertTrue(self.store.verify(result["artifacts"]["observed_surface"]))
        self.assertEqual(inspect_job(self.store, result["job"])["integrity_problems"], [])
        self.assertFalse((Path(result["manifest"]).parent / "work").exists())
        self.assertFalse(self.store.path.exists())
        cleaned = self.store.artifact_path(result["artifacts"]["cleaned_mesh"])
        cleaned.write_text("changed")
        self.assertIn("cleaned_mesh", inspect_job(self.store, result["job"])["integrity_problems"])

    def test_native_failure_persists_diagnostics_but_not_partial_products(self):
        result = reconstruct(self.store, RUN, "face-001", FakeTools(fail_at="dense-stereo"), self.policy)
        self.assertEqual(result["status"], "failed")
        self.assertIn("cameras", result["artifacts"])
        self.assertNotIn("reconstructed_mesh", result["artifacts"])
        self.assertFalse(list(Path(result["manifest"]).parent.rglob("partial.tmp")))
        self.assertGreater(result["cleanup"]["removed_files"], 0)

    def test_explicit_poc_relaxes_native_constraints_and_retains_policy_without_certifying_likeness(self):
        policy = replace(self.policy, mapper_init_min_inliers=20, mapper_init_min_tri_angle=0.5,
                         mapper_abs_pose_min_inliers=15, stereo_min_consistent=1, fusion_min_pixels=2)
        tools = FakeTools()
        with patch.object(tools, "run", wraps=tools.run) as native:
            result = reconstruct(self.store, RUN, "face-001", tools, policy)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["policy"], asdict(policy))
        self.assertFalse(result["likeness_verified"])
        self.assertFalse(result["production_ready"])
        options = {call.args[1]: dict(zip(call.args[2][1::2], call.args[2][2::2]))
                   for call in native.call_args_list if call.args[0] == "colmap"}
        self.assertEqual(options["sparse-mapping"]["--Mapper.init_min_num_inliers"], "20")
        self.assertEqual(options["sparse-mapping"]["--Mapper.init_min_tri_angle"], "0.5")
        self.assertEqual(options["sparse-mapping"]["--Mapper.abs_pose_min_num_inliers"], "15")
        self.assertEqual(options["dense-stereo"]["--PatchMatchStereo.filter_min_num_consistent"], "1")
        self.assertEqual(options["surface-fusion"]["--StereoFusion.min_num_pixels"], "2")

    def test_incomplete_blender_export_is_not_committed_as_cleaned_mesh(self):
        result = reconstruct(self.store, RUN, "face-001", FakeTools(missing_render=True), self.policy)
        self.assertEqual(result["status"], "failed")
        self.assertIn("reconstructed_mesh", result["artifacts"])
        self.assertNotIn("cleaned_mesh", result["artifacts"])
        self.assertFalse((Path(result["manifest"]).parent / "blender").exists())

    def test_cleanup_protects_portraits_referenced_only_by_generation(self):
        inputs = prepare_inputs(self.store, RUN, "face-001", self.policy)
        referenced = [self.store.resolve(e["portrait"]["path"]) for e in inputs["inputs"]["entries"]]
        self.run_path.unlink()
        self.store.resolve(PROFILE).unlink()
        cleanup_alias(self.store)
        self.assertTrue(all(path.is_file() for path in referenced))

    def test_missing_input_manifest_is_reported_as_integrity_problem(self):
        result = reconstruct(self.store, RUN, "face-001", FakeTools(connected=False), self.policy)
        self.store.artifact_path(result["input_view"]).unlink()
        self.assertIn("input_view", inspect_job(self.store, result["job"])["integrity_problems"])


class CameraAndToolTests(unittest.TestCase):
    def test_camera_export_restores_crop_offsets_and_reprojects_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cameras.txt").write_text("1 SIMPLE_PINHOLE 28 28 100 14 14\n")
            (root / "points3D.txt").write_text("8 0 0 2 128 128 128 0 1 0\n")
            # 90-degree world-to-camera rotation about Z; translated camera.
            q = 2**-0.5
            (root / "images.txt").write_text(
                f"1 {q} 0 0 {q} 0.1 0.2 0 1 face.png\n19 24 8\n")
            inputs = {"inputs": {"entries": [{"image_name": "face.png", "source_id": "original",
                                             "face_id": "face", "size": [28, 28],
                                             "portrait_bounds": [10, 20, 38, 48]}]}}
            result = read_model(root, inputs)
            camera = result["cameras"]["original"]
            self.assertAlmostEqual(camera["reprojection_error_px"], 0)
            self.assertEqual(camera["intrinsics"], [[100, 0, 24], [0, 100, 34], [0, 0, 1]])
            self.assertAlmostEqual(camera["world_to_camera"][0][1], -1)
            (root / "points3D.txt").write_text("8 nan 0 2 0 0 0 0 1 0\n")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                read_model(root, inputs)

    def test_per_corner_uv_seams_survive_ply_to_obj_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "mesh.ply", root / "head.obj"
            source.write_text(TEXTURED_TRIANGLE)
            result = textured_ply_to_obj(source, target)
            self.assertEqual(result["faces"], 1)
            self.assertIn("f 1/1 2/2 3/3", target.read_text())
            self.assertIn("vt 0.5 1", target.read_text())
            source.write_text(TEXTURED_TRIANGLE.replace("3 0 1 2 6", "3 0 1 9 6"))
            with self.assertRaisesRegex(ValueError, "Invalid textured face"):
                textured_ply_to_obj(source, target)

    def test_view_graph_counts_connected_components_not_just_number_of_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "database.db"
            create_database(database, 6, False)
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.executemany("INSERT INTO two_view_geometries VALUES (?,?)",
                                       [(COLMAP_PAIR_BASE + 2, 20),
                                        (COLMAP_PAIR_BASE*3 + 4, 25),
                                        (COLMAP_PAIR_BASE*4 + 5, 30)])
            result = matching_summary(database)
            self.assertEqual(result["largest_connected_component"], 3)
            self.assertEqual(result["connected_component_sizes"], [3, 2])
            self.assertEqual(result["images_with_verified_matches"], 5)

    def test_native_failure_bounds_log_and_timeout_has_no_silent_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = NativeTools(root, 1)

            def failing(args, **kwargs):
                kwargs["stdout"].write(b"x" * (MAX_LOG_BYTES + 100))
                return subprocess.CompletedProcess(args, 9)

            with patch("ailias.generation.tools.subprocess.run", side_effect=failing):
                with self.assertRaisesRegex(ToolError, "exit_9"):
                    tools.run("colmap", "failure", ["mapper"], root)
            self.assertLess((root / "logs" / "failure.log").stat().st_size, MAX_LOG_BYTES + 100)
            with patch("ailias.generation.tools.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("colmap", 1)):
                with self.assertRaisesRegex(ToolError, "TimeoutExpired"):
                    tools.run("colmap", "timeout", ["mapper"], root)

    def test_policy_rejects_unbounded_or_nonfinite_work(self):
        for values in ({"max_images": 65}, {"min_registered_fraction": 0},
                       {"max_reprojection_error_px": float("nan")},
                       {"poisson_depth": 15}, {"gpu_index": -1},
                       {"mapper_init_min_tri_angle": 180}, {"fusion_min_pixels": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(GenerationPolicy(), **values)


if __name__ == "__main__":
    unittest.main()
