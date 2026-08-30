import tempfile
import unittest
from pathlib import Path

import torch

from aigc_detector.e4a import (
    FEATURE_BLOCKS,
    MODALITY_SUBSETS,
    modality_indices,
    rank_ablation_rows,
    require_validation_selection,
    select_modalities,
    validation_selection_tensors,
    write_summaries,
)


class E4aTests(unittest.TestCase):
    def test_named_feature_slices_and_all_subsets(self) -> None:
        self.assertEqual({name: block.stop - block.start for name, block in FEATURE_BLOCKS.items()}, {
            "clip": 512, "laplacian": 1280, "fft": 1280,
        })
        self.assertEqual(len(MODALITY_SUBSETS), 7)
        expected = {
            "clip": 512, "laplacian": 1280, "fft": 1280, "laplacian+fft": 2560,
            "clip+laplacian": 1792, "clip+fft": 1792, "clip+laplacian+fft": 3072,
        }
        features = torch.arange(3072.0).view(1, -1)
        for subset, width in expected.items():
            selected = select_modalities(features, subset)
            self.assertEqual(selected.shape[1], width)
            self.assertEqual(len(set(modality_indices(subset).tolist())), width)
        self.assertTrue(torch.equal(select_modalities(features, "fft")[0], torch.arange(1792.0, 3072.0)))

    def test_wrong_width_and_test_selection_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "3072"):
            select_modalities(torch.zeros(2, 3071), "clip")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_selection("test")
        val_x, val_y = validation_selection_tensors({
            "val_features": torch.tensor([[1.0]]), "val_labels": torch.tensor([0.0]),
            "test_features": torch.tensor([[999.0]]), "test_labels": torch.tensor([1.0]),
        })
        self.assertEqual(val_x.item(), 1.0)
        self.assertEqual(val_y.item(), 0.0)

    def test_summary_ranking_uses_validation_metrics(self) -> None:
        template = {
            "feature_dimension": 1, "trainable_parameter_count": 2,
            "selection_split": "validation", "test_rows_used": False,
            "worst_condition": "resize_x0.25", "mean_transformed_roc_auc": 0.7,
            "worst_transformed_roc_auc": 0.6, "clean_false_positive_rate": 0.1,
            "mean_transformed_false_positive_rate": 0.2, "temperature": 1.0, "threshold": 0.5,
            "per_condition_validation": {"resize_x0.25": {"balanced_accuracy": 0.5}},
        }
        rows = [
            {**template, "modality_subset": "a", "clean_validation_balanced_accuracy": 0.9,
             "mean_transformed_validation_balanced_accuracy": 0.7,
             "worst_transformed_validation_balanced_accuracy": 0.6},
            {**template, "modality_subset": "b", "clean_validation_balanced_accuracy": 0.8,
             "mean_transformed_validation_balanced_accuracy": 0.8,
             "worst_transformed_validation_balanced_accuracy": 0.7},
        ]
        self.assertEqual(rank_ablation_rows(rows)[0]["modality_subset"], "b")
        with tempfile.TemporaryDirectory() as tmp:
            write_summaries(rows, Path(tmp))
            text = (Path(tmp) / "ablation_summary.json").read_text()
            self.assertIn('"selection_split": "validation"', text)
            self.assertIn('"test_rows_used": false', text)


if __name__ == "__main__":
    unittest.main()
