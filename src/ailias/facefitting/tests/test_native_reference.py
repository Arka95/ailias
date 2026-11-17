"""Exercise measured iris colour and render-to-reference correction with synthetic pixels."""

import unittest

import numpy as np

from ailias.facefitting.native_reference import iris_tint_adjustment


class IrisReferenceTests(unittest.TestCase):
    def test_linear_correction_preserves_current_tint_and_reports_clipping(self):
        result = iris_tint_adjustment(
            {"median_linear_rgb": [.1, .4, .02]},
            {"median_linear_rgb": [.2, .1, .1]}, [.8, .5, .25])
        np.testing.assert_allclose(result["eyes"]["global_tint_linear"], [.4, 1, .05])
        self.assertEqual(result["clipped_channels"], [1])
        self.assertTrue(result["render_verification_required"])
        self.assertFalse(result["intrinsic_colour_calibrated"])

    def test_missing_render_colour_cannot_produce_a_tint(self):
        with self.assertRaises(ValueError):
            iris_tint_adjustment({"median_linear_rgb": [.1]*3},
                                 {"median_linear_rgb": [0]*3}, [1]*3)


if __name__ == "__main__":
    unittest.main()
