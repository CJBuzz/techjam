import json
import tempfile
import unittest
from pathlib import Path

import torch

from aigc_detector.e1b import (
    PairedFeatureStore,
    completed_result,
    contribution_statistics,
    rank_results,
    require_validation_selection,
    weighted_mean,
)


class E1bTests(unittest.TestCase):
    def test_external_weight_endpoints_and_intermediate_scaling(self) -> None:
        losses = torch.tensor([2.0, 4.0])
        external = torch.tensor([False, True])
        self.assertEqual(weighted_mean(losses, external, 0.0).item(), 2.0)
        self.assertEqual(weighted_mean(losses, external, 1.0).item(), losses.mean().item())
        self.assertAlmostEqual(weighted_mean(losses, external, 0.5).item(), 8 / 3, places=6)
        stats = contribution_statistics(10, 10, 6, 0.5)
        self.assertAlmostEqual(stats["external_effective_fraction"], 1 / 3)

    def test_paired_store_preserves_source_and_repeat_metadata(self) -> None:
        store = PairedFeatureStore(
            torch.tensor([[1.0], [11.0]]), torch.tensor([0.0, 0.0]), ["clean", "jpeg"], 1,
            torch.tensor([[2.0], [12.0]]), torch.tensor([1.0, 1.0]), ["clean", "jpeg"], 1,
        )
        features, labels, groups, external = store.batch(torch.tensor([1, 0]))
        self.assertEqual(features.flatten().tolist(), [2.0, 1.0, 12.0, 11.0])
        self.assertEqual(labels.tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(groups, ["clean", "clean", "jpeg", "jpeg"])
        self.assertEqual(external.tolist(), [True, False, True, False])

    def test_validation_ranking_ignores_external_metrics_and_test(self) -> None:
        base = {
            "status": "succeeded", "selection_split": "validation", "test_rows_used": False,
            "clean_validation_balanced_accuracy": 0.90,
            "mean_transformed_validation_balanced_accuracy": 0.70,
            "worst_transformed_validation_balanced_accuracy": 0.60,
        }
        rows = [
            {**base, "external_weight": 0.0, "external_roc_auc": 0.99},
            {**base, "external_weight": 0.5, "clean_validation_balanced_accuracy": 0.895,
             "mean_transformed_validation_balanced_accuracy": 0.72,
             "worst_transformed_validation_balanced_accuracy": 0.65,
             "external_roc_auc": 0.01},
            {**base, "external_weight": 1.0, "clean_validation_balanced_accuracy": 0.88,
             "worst_transformed_validation_balanced_accuracy": 0.99},
        ]
        ranked = rank_results(rows)
        self.assertEqual(next(row for row in ranked if row["external_weight"] == 0.5)["primary_track5_rank"], 1)
        self.assertIsNone(next(row for row in ranked if row["external_weight"] == 1.0)["primary_track5_rank"])
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_selection("test")

    def test_resume_requires_successful_safe_result_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "model.pt").write_bytes(b"checkpoint")
            result = {
                "status": "succeeded", "external_weight": 0.2,
                "selection_split": "validation", "test_rows_used": False,
            }
            (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(completed_result(directory, 0.2), result)
            self.assertIsNone(completed_result(directory, 0.1))


if __name__ == "__main__":
    unittest.main()
