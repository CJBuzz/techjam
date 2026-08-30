import unittest

import numpy as np
import torch
from PIL import Image

from aigc_detector.data import average_view_logits, test_time_views


class MildTtaTests(unittest.TestCase):
    def test_mild3_views_are_exact_and_deterministic(self) -> None:
        image = Image.fromarray(np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3), "RGB")
        first = test_time_views(image, "mild3", 42, "sample")
        second = test_time_views(image, "mild3", 42, "sample")
        self.assertEqual(len(first), 3)
        self.assertTrue(all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first, second)))
        self.assertTrue(np.array_equal(np.asarray(first[0]), np.asarray(image)))

    def test_logits_are_averaged_before_thresholding(self) -> None:
        logits = torch.tensor([[-4.0, 2.0, 2.0]])
        probability = torch.sigmoid(average_view_logits(logits) / 2.0)
        self.assertAlmostEqual(float(average_view_logits(logits)), 0.0)
        self.assertFalse(bool(probability >= 0.51))
        self.assertTrue(bool(torch.sigmoid(logits / 2.0).mean(-1) >= 0.51))


if __name__ == "__main__":
    unittest.main()
