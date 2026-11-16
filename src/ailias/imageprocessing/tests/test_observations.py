import copy
import unittest

import numpy as np
from PIL import Image, ImageDraw

from ailias.imageprocessing.observations import align_head, assess_lighting, face_lighting


class ObservationTests(unittest.TestCase):
    def test_alignment_levels_eyes_and_roundtrips_all_landmarks_without_changing_original(self):
        image = Image.new("RGBA", (260, 320), (0, 0, 0, 0))
        ImageDraw.Draw(image).ellipse((20, 10, 240, 300), fill=(160, 110, 75, 255))
        points = np.array([[80, 95], [170, 125], [120, 160], [85, 200], [150, 215]])
        before = image.tobytes()
        aligned, record = align_head(image, points.tolist())
        self.assertEqual(image.tobytes(), before)
        mapped = np.asarray(record["landmarks_5_aligned"])
        self.assertAlmostEqual(mapped[0, 1], mapped[1, 1])
        np.testing.assert_allclose(mapped[:2].mean(0), [256, 204.8])
        homogeneous = np.column_stack((mapped, np.ones(5)))
        restored = homogeneous @ np.asarray(record["aligned_to_portrait"]).T
        np.testing.assert_allclose(restored[:, :2], points, atol=1e-10)
        box = aligned.getchannel("A").getbbox()
        self.assertGreater(box[0], 0)
        self.assertGreater(box[1], 0)
        self.assertLess(box[2], aligned.width)
        self.assertLess(box[3], aligned.height)
        pixels = np.asarray(aligned)
        visible = pixels[:, :, 3] > 0
        np.testing.assert_allclose(pixels[visible][:, :3], np.tile([160, 110, 75], (visible.sum(), 1)), atol=1)

    def test_alignment_abstains_instead_of_inventing_eye_positions(self):
        image = Image.new("RGBA", (100, 100), (160, 100, 60, 255))
        for points in ([], [[10, 10]]*5, [[float("nan"), 2]]*5):
            aligned, record = align_head(image, points)
            self.assertIsNone(aligned)
            self.assertEqual(record["status"], "unavailable")

    def lighting(self, color):
        image = Image.new("RGB", (128, 128), color)
        return face_lighting(image, Image.new("L", image.size, 1),
                             Image.new("L", image.size, 255), [0, 0, 128, 128])

    def test_relative_lighting_identifies_a_colour_outlier_without_changing_skin_or_neutralizing_it(self):
        records = {
            str(index): {
                "lighting": self.lighting(color), "view": {"category": "front"},
                "status": "ready", "subject_scope": "subject", "quality_warnings": [],
            }
            for index, color in enumerate(((160, 105, 75), (170, 112, 80),
                                           (180, 118, 85), (60, 60, 180)))
        }
        before = copy.deepcopy(records)
        result = assess_lighting(records)
        self.assertIn("subject", result["references"])
        for index in ("0", "1", "2"):
            self.assertNotIn("relative_subject_colour_outlier", records[index]["lighting"]["warnings"])
        self.assertIn("relative_subject_colour_outlier", records["3"]["lighting"]["warnings"])
        self.assertGreater(records["3"]["lighting"]["baseline_penalty"], records["0"]["lighting"]["baseline_penalty"])
        for key in records:
            self.assertEqual(records[key]["lighting"]["chromaticity"], before[key]["lighting"]["chromaticity"])
        for key, record in records.items():
            record["subject_scope"] = key
        assess_lighting(records)
        for record in records.values():
            self.assertNotIn("subject_chromaticity_distance", record["lighting"])
            self.assertNotIn("relative_subject_colour_outlier", record["lighting"]["warnings"])

    def test_lighting_does_not_compare_different_subjects_or_invent_missing_references(self):
        records = {
            str(index): {"lighting": self.lighting(color), "view": {"category": "front"},
                         "status": "ready", "subject_scope": str(index), "quality_warnings": []}
            for index, color in enumerate(((160, 105, 75), (60, 60, 180)))
        }
        self.assertFalse(assess_lighting(records)["references"])
        self.assertTrue(all(not record["lighting"]["warnings"] for record in records.values()))

    def test_skin_shadow_evidence_uses_only_visible_skin_and_exposes_uncertainty(self):
        image = Image.new("RGB", (128, 128), (180, 120, 90))
        ImageDraw.Draw(image).rectangle((0, 0, 63, 127), fill=(45, 30, 22))
        parts, alpha = Image.new("L", image.size, 1), Image.new("L", image.size, 255)
        evidence = face_lighting(image, parts, alpha, [0, 0, 128, 128])
        self.assertGreater(evidence["left_right_luma_ratio"], 3.9)
        parts.paste(0, (0, 0, 128, 128))
        evidence = face_lighting(image, parts, alpha, [0, 0, 128, 128])
        self.assertEqual(evidence["status"], "insufficient_visible_skin")
        self.assertNotIn("chromaticity", evidence)
