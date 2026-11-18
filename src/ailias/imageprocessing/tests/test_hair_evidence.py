"""Region darkness remains explicitly heuristic and fails cleanly for unsupported inputs."""

import unittest

import numpy as np

from ailias.imageprocessing.hair_evidence import REGIONS, facial_hair_evidence


class FacialHairEvidenceTests(unittest.TestCase):
    def fixture(self):
        rgb = np.full((100, 100, 3), 180, dtype=np.uint8)
        points = np.full((478, 2), 50., dtype=float)
        for indices in REGIONS.values():
            angle = np.linspace(0, 2*np.pi, len(indices), endpoint=False)
            points[list(indices)] = np.column_stack((50+30*np.cos(angle), 50+30*np.sin(angle)))
        return rgb, points, np.ones((100, 100), dtype=np.uint8), [.45, .45, .45]

    def test_dark_region_is_an_estimate_not_a_new_segmentation_class(self):
        rgb, points, labels, skin = self.fixture()
        rgb[30:70, 30:70] = 30
        result = facial_hair_evidence(rgb, points, labels, skin)
        self.assertFalse(result["distinct_parser_class"])
        self.assertIn("shadows", result["confidence"])
        self.assertTrue(result["regions"]["beard"]["presence_estimate"])

    def test_uniform_visible_skin_does_not_produce_beard_evidence(self):
        result = facial_hair_evidence(*self.fixture())
        self.assertFalse(result["regions"]["beard"]["presence_estimate"])
        self.assertEqual(result["regions"]["beard"]["dark_fraction"], 0)

    def test_occluded_or_missing_skin_remains_unresolved(self):
        rgb, points, labels, skin = self.fixture()
        labels[:] = 0
        result = facial_hair_evidence(rgb, points, labels, skin)
        self.assertEqual(result["regions"]["beard"]["status"], "unresolved")

    def test_nonfinite_landmarks_are_rejected(self):
        rgb, points, labels, skin = self.fixture()
        points[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite 478"):
            facial_hair_evidence(rgb, points, labels, skin)
