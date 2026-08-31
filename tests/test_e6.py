import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigc_detector.data import RobustTransform
from aigc_detector.experiments.e6 import (
    ScaleTransform,
    add_scale_objective,
    require_validation_selection,
    scale_cache_valid,
    scale_consistency_loss,
)


class E6Tests(unittest.TestCase):
    def test_resize_matches_official_semantics(self) -> None:
        image = Image.fromarray(np.arange(31 * 47 * 3, dtype=np.uint8).reshape(31, 47, 3), "RGB")
        expected = RobustTransform._apply_one(image, "resize", 0.5)
        actual = ScaleTransform(0.5)(image)
        self.assertTrue(np.array_equal(np.asarray(actual), np.asarray(expected)))
        self.assertEqual(actual.size, image.size)

    def test_asymmetric_teacher_is_detached_and_weights_apply(self) -> None:
        logits = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True)
        weights = torch.tensor([0.5, 1.0, 1.5])
        loss = scale_consistency_loss(logits, "logit_asymmetric", weights)
        expected = (0.5 * 1 + 1.0 * 4 + 1.5 * 9) / 3.0
        self.assertAlmostEqual(float(loss.detach()), expected)
        loss.backward()
        self.assertEqual(float(logits.grad[0]), 0.0)
        self.assertTrue(bool((logits.grad[1:] != 0).all()))

    def test_lambda_zero_preserves_old_objective_exactly(self) -> None:
        base = torch.tensor(2.5, requires_grad=True)
        scale = torch.tensor(9.0, requires_grad=True)
        combined = add_scale_objective(base, scale, 0.0)
        self.assertEqual(float(combined.detach()), float(base.detach()))
        combined.backward()
        self.assertEqual(float(base.grad), 1.0)
        self.assertEqual(float(scale.grad), 0.0)

    def test_cache_compatibility_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            manifest = {"row_count": 2, "scale": 0.5}
            torch.save({"features": torch.zeros(2, 3), "manifest": manifest}, path)
            self.assertTrue(scale_cache_valid(path, manifest))
            self.assertFalse(scale_cache_valid(path, {**manifest, "scale": 0.25}))

    def test_final_test_cannot_select_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_selection("test")


if __name__ == "__main__":
    unittest.main()
