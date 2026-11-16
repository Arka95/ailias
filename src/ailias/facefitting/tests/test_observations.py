import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ailias.facefitting.observations import observe
from ailias.storage import Store


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, rgb):
        return {"faces": self.faces, "blendshapes": [], "transforms": []}


class ObservationTests(unittest.TestCase):
    def test_source_coordinates_are_recovered_and_background_points_not_used(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder), "test")
            store.root.mkdir()
            image = Image.new("RGBA", (100, 100), (100, 100, 100, 255))
            image.putpixel((0, 0), (0, 0, 0, 0))
            path = store.root / "head.png"
            image.save(path)
            entry = {"face_id": "face", "source_id": "source", "analysis": {
                "aligned_cutout": store.reference_artifact(path), "alignment": {
                    "status": "ready", "aligned_to_source": [[2, 0, 10], [0, 2, 20], [0, 0, 1]],
                    "source_to_aligned": [[.5, 0, -5], [0, .5, -10], [0, 0, 1]]}}}
            landmarks = np.tile([.5, .5, 0], (478, 1))
            landmarks[0] = [0, 0, 0]
            landmarks[33, 0], landmarks[263, 0] = .3, .7
            result = observe(store, entry, [1000, 800], FakeDetector([landmarks]))
            self.assertEqual(result["status"], "ready")
            np.testing.assert_allclose(result["source_xy"][1], [110, 120])
            self.assertFalse(result["on_head"][0])
            self.assertFalse(result["landmark_depth_is_measured"])
            self.assertEqual(observe(store, entry, [1000, 800], FakeDetector([]))["status"], "review")
            with self.assertRaises(ValueError):
                entry["analysis"]["alignment"]["source_to_aligned"][0][0] = .6
                observe(store, entry, [1000, 800], FakeDetector([landmarks]))


if __name__ == "__main__":
    unittest.main()
