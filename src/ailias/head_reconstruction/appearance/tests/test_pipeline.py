"""Verify appearance approval and native-portrait semantic coordinates with isolated workers."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from ailias.head_reconstruction.appearance.pipeline import (
    approved_fit,
    generate_head_appearance,
)
from ailias.head_reconstruction.evidence.semantic_parsing_worker import main
from ailias.imageprocessing.faceparsing import DIRECTORY
from ailias.storage import Store, atomic_json


class AppearanceTests(unittest.TestCase):
    def test_partial_semantic_photo_arrays_are_removed_when_owned_worker_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary), "subject")
            inputs = store.resolve("inputs.json")
            fake_photo = store.resolve("photo.txt")
            fake_photo.parent.mkdir(parents=True)
            fake_photo.write_text("synthetic input, never decoded")
            reference = store.reference_artifact(fake_photo)
            atomic_json(inputs, {"inputs": {"entries": [{
                "face_id": "face", "image": reference, "source": reference,
                "foreground_alpha": reference}]}})
            cameras, observations = store.resolve("cameras.json"), store.resolve("observations.json")
            atomic_json(cameras, {"views": []})
            atomic_json(observations, {"faces": []})
            fitted = {"inputs": store.reference_artifact(inputs), "assets": {}, "artifacts": {
                "mesh": reference, "cameras": store.reference_artifact(cameras),
                "observations": store.reference_artifact(observations)}}
            atomic_json(store.resolve("facefitting\\runs\\job\\manifest.json"), fitted)
            touched = []

            def failing_worker(name, operation, args, directory):
                path = directory / "photo-00.npz"
                path.write_bytes(b"synthetic partial photo array")
                touched.append(path)
                raise ValueError("synthetic native failure")

            with patch("ailias.head_reconstruction.appearance.pipeline.approved_fit", return_value=(fitted, reference)), \
                    patch("ailias.head_reconstruction.appearance.pipeline.choose_texture_views",
                          return_value=[{"face_id": "face"}]):
                result = generate_head_appearance(store, "job", {"device": "cuda:0", "photo_size": 256, "max_views": 1},
                                        SimpleNamespace(), SimpleNamespace(paths={}, run=failing_worker))
            self.assertEqual(result["status"], "failed")
            self.assertTrue(touched)
            self.assertFalse(touched[0].exists())
            self.assertTrue(store.verify(reference))

    def test_parser_uses_portrait_box_once_even_when_original_crop_has_large_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary), "subject")
            output = store.resolve("appearance-test")
            output.mkdir(parents=True)
            image, alpha = output / "rgb.png", output / "alpha.png"
            Image.new("RGB", (120, 80), (150, 100, 80)).save(image)
            Image.new("L", (120, 80), 255).save(alpha)
            entry = {"face_id": "test", "box_xywh": [30, 20, 40, 40],
                     "portrait_bounds": [1000, 2000, 1120, 2080],
                     "image": store.reference_artifact(image), "source": store.reference_artifact(image),
                     "foreground_alpha": store.reference_artifact(alpha)}
            atomic_json(store.resolve(str(Path(DIRECTORY) / "provenance.json")),
                        {"license_artifact": entry["image"]})
            request = output / "request.json"
            atomic_json(request, {"data_root": temporary, "alias": "subject", "output": str(output),
                                  "device": "cuda:0", "photo_size": 60, "photos": [{"entry": entry}]})
            with patch("ailias.head_reconstruction.evidence.semantic_parsing_worker.HeadParser") as factory:
                parser = factory.return_value
                parser.provenance = {}
                parser.parse.return_value = Image.new("L", (60, 40), 1)
                main(request)
                self.assertEqual(parser.parse.call_args.args[1], [15, 10, 20, 20])
                parser.close.assert_called_once()
            result = json.loads((output / "semantics.json").read_text())
            self.assertEqual(result["photos"][0]["portrait_scale"], [.5, .5])

    def test_approval_is_rejected_if_neutral_mesh_reference_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary), "subject")
            approval = store.resolve("facefitting\\reviews\\job-clay-approval.json")
            atomic_json(approval, {"scope": "neutral_clay_geometry", "decision": "approved",
                                  "mesh": {"path": "old.obj", "sha256": "old"}})
            fitted = {"integrity_problems": [], "artifacts": {
                "mesh": {"path": "new.obj", "sha256": "new"}}}
            with patch("ailias.head_reconstruction.appearance.pipeline.inspect_head_fit", return_value=fitted):
                with self.assertRaisesRegex(ValueError, "exact neutral geometry"):
                    approved_fit(store, "job")


if __name__ == "__main__":
    unittest.main()
