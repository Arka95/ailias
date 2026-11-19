"""Shared projection retains current source inputs and cleans owned temporary arrays."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ailias.head_reconstruction.appearance.photo_texture_baking import (
    project_photo_evidence,
    write_texture_atlas,
)
from ailias.head_reconstruction.geometry.tests.fitting_fixtures import make_head_fit
from ailias.storage import Store


class PhotoTextureBakingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name), "subject")
        self.root = self.store.root
        self.root.mkdir()
        image, mask = self.root / "sample.png", self.root / "regions.png"
        Image.new("RGB", (8, 8), (100, 80, 60)).save(image)
        Image.new("L", (8, 8), 1).save(mask)
        self.source = self.store.reference_artifact(image)
        self.evidence = {
            "frames": [
                {"face_id": "frame", "image": self.source, "source_size": [8, 8]}
            ],
            "faces": {
                "frame": {
                    "regions": self.store.reference_artifact(mask),
                    "features": {
                        "regions": {
                            "skin": {
                                "status": "observed",
                                "median_linear_rgb": [0.2, 0.15, 0.1],
                            }
                        }
                    },
                }
            },
        }
        self.fitted = make_head_fit(["frame"], error=0.02)
        self.views = [{"face_id": "frame", "view": {"category": "front", "yaw_deg": 0}}]

    def test_projection_reads_current_arrays_and_removes_them_after_success_or_failure(
        self,
    ):
        for fail in (False, True):
            with self.subTest(fail=fail):

                def bake(surface, identity, mapping, selected, entries, photos, device):
                    with np.load(photos[0]["data"], allow_pickle=False) as data:
                        self.assertEqual(data["rgb"].shape, (8, 8, 3))
                        np.testing.assert_array_equal(data["labels"], np.ones((8, 8)))
                    if fail:
                        raise RuntimeError("projection failed")
                    return {"current_projection": True}

                with patch(
                    "ailias.head_reconstruction.appearance.cuda_texture_projection.bake",
                    side_effect=bake,
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "projection failed"):
                            project_photo_evidence(
                                self.store,
                                self.root,
                                object(),
                                np.zeros(1),
                                (),
                                self.fitted,
                                self.views,
                                self.evidence,
                                "cuda:0",
                            )
                    else:
                        result, selected = project_photo_evidence(
                            self.store,
                            self.root,
                            object(),
                            np.zeros(1),
                            (),
                            self.fitted,
                            self.views,
                            self.evidence,
                            "cuda:0",
                        )
                        self.assertTrue(result["current_projection"])
                        self.assertEqual(selected[0]["face_id"], "frame")
                self.assertFalse(list(self.root.glob("photo-texture-*.npz")))
                self.assertTrue(self.store.verify(self.source))

    def test_empty_photo_coverage_cannot_write_a_successful_atlas(self):
        mapping = (np.array([0]), np.array([0]), np.array([0]), np.ones((1, 3)) / 3)
        baked = {
            "skin_color": np.ones(3) * 0.4,
            "coverage": np.array([False]),
            "color": np.ones((1, 3)) * 0.4,
        }
        with self.assertRaisesRegex(RuntimeError, "No photographed colour"):
            write_texture_atlas(
                self.root,
                mapping,
                baked,
                8,
                texture_name="atlas.png",
                coverage_name="coverage.png",
            )
        self.assertFalse((self.root / "atlas.png").exists())
