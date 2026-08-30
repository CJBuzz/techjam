import tempfile
import unittest
from pathlib import Path

import torch

from aigc_detector.e1c import (
    ENSEMBLE_MODES,
    aggregate_checkpoint_views,
    blend_calibrated_logits,
    load_or_build_logit_cache,
    rank_results,
    require_validation_selection,
)


class E1cTests(unittest.TestCase):
    def test_endpoint_interpolation_and_calibration(self) -> None:
        e0 = torch.tensor([4.0, -2.0])
        e1 = torch.tensor([-3.0, 6.0])
        self.assertTrue(torch.equal(blend_calibrated_logits(e0, 2.0, e1, 3.0, 1.0), e0 / 2.0))
        self.assertTrue(torch.equal(blend_calibrated_logits(e0, 2.0, e1, 3.0, 0.0), e1 / 3.0))
        expected = 0.25 * (e0 / 2.0) + 0.75 * (e1 / 3.0)
        self.assertTrue(torch.equal(blend_calibrated_logits(e0, 2.0, e1, 3.0, 0.25), expected))

    def test_raw_and_existing_mild3_are_the_only_modes(self) -> None:
        self.assertEqual(ENSEMBLE_MODES, {
            "raw": ("clean",), "mild3": ("clean", "jpeg_q90", "resize_x0.5")
        })
        views = torch.tensor([[1.0, 4.0, 7.0]])
        self.assertEqual(float(aggregate_checkpoint_views(views, "raw")), 1.0)
        self.assertEqual(float(aggregate_checkpoint_views(views, "mild3")), 4.0)

    def test_validation_ranking_excludes_external_metrics(self) -> None:
        base = {
            "mode": "raw", "clean_validation_balanced_accuracy": 0.90,
            "mean_transformed_validation_balanced_accuracy": 0.70,
            "worst_transformed_validation_balanced_accuracy": 0.60,
            "external_roc_auc": 0.99,
        }
        rows = [
            {**base, "alpha": 1.0},
            {**base, "alpha": 0.5, "mean_transformed_validation_balanced_accuracy": 0.72,
             "worst_transformed_validation_balanced_accuracy": 0.65, "external_roc_auc": 0.01},
            {**base, "alpha": 0.0, "clean_validation_balanced_accuracy": 0.88,
             "worst_transformed_validation_balanced_accuracy": 0.99},
        ]
        ranked = rank_results(rows)
        self.assertEqual(next(row for row in ranked if row["alpha"] == 0.5)["rank"], 1)
        self.assertIsNone(next(row for row in ranked if row["alpha"] == 0.0)["rank"])
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_selection("test")

    def test_cached_logits_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            calls = []
            def builder():
                calls.append(1)
                return {"raw_view_logits_e0": torch.ones(1)}
            load_or_build_logit_cache(path, {"split": "validation"}, builder)
            load_or_build_logit_cache(path, {"split": "validation"}, builder)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
