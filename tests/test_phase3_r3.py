import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aigc_detector.phase3.artifacts import REQUIRED_FILES
from aigc_detector.phase3.data import ManifestRecord
from aigc_detector.phase3.r3 import (
    CONSISTENCY_CONFIGS,
    PairedDataset,
    asymmetric_feature_distance,
    asymmetric_prediction_divergence,
    paired_loss,
    select_promotion_setting,
)


class DummyProcessor:
    def __call__(self, images, size, return_tensors):
        array = np.asarray(images.resize((size["width"], size["height"])), dtype=np.float32) / 255
        return {"pixel_values": torch.tensor(array).permute(2, 0, 1).unsqueeze(0)}


class R3Tests(unittest.TestCase):
    def test_prediction_teacher_is_detached(self) -> None:
        clean = torch.tensor([0.2, -0.4], requires_grad=True)
        corrupt = torch.tensor([0.0, 0.1], requires_grad=True)
        loss = asymmetric_prediction_divergence(clean, corrupt); loss.backward()
        self.assertIsNone(clean.grad); self.assertIsNotNone(corrupt.grad)

    def test_feature_teacher_is_detached(self) -> None:
        clean = torch.randn(2, 4, requires_grad=True); corrupt = torch.randn(2, 4, requires_grad=True)
        asymmetric_feature_distance(clean, corrupt).backward()
        self.assertIsNone(clean.grad); self.assertIsNotNone(corrupt.grad)

    def test_zero_lambdas_exactly_disable_consistency(self) -> None:
        labels = torch.tensor([0.0, 1.0]); clean = torch.tensor([0.1, 0.2]); corrupt = torch.tensor([-0.1, 0.3])
        clean_features, corrupt_features = torch.randn(2, 3), torch.randn(2, 3)
        losses = paired_loss(clean, corrupt, clean_features, corrupt_features, labels, 0.0, 0.0)
        expected = F.binary_cross_entropy_with_logits(clean, labels) + F.binary_cross_entropy_with_logits(corrupt, labels)
        self.assertTrue(torch.equal(losses["total"], expected))
        self.assertEqual(float(losses["prediction"]), 0.0); self.assertEqual(float(losses["feature"]), 0.0)

    def test_loss_coefficients_are_positive_penalties(self) -> None:
        labels = torch.tensor([0.0, 1.0]); clean = torch.tensor([0.2, -0.1]); corrupt = torch.tensor([-0.2, 0.3])
        clean_features, corrupt_features = torch.randn(2, 3), torch.randn(2, 3)
        losses = paired_loss(clean, corrupt, clean_features, corrupt_features, labels, 0.15, 0.05)
        expected = losses["classification"] + 0.15 * losses["prediction"] + 0.05 * losses["feature"]
        self.assertTrue(torch.allclose(losses["total"], expected))

    def test_paired_dataset_returns_one_shared_label_and_no_condition_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"; Image.new("RGB", (12, 12), "red").save(path)
            record = ManifestRecord(path=str(path), label=1, split="train", source="toy", unique_id="one")
            dataset = PairedDataset([record], DummyProcessor(), 8, 42, "compound_curriculum", 2)
            clean, corrupt, label, returned_path = dataset[0]
            self.assertEqual(clean.shape, corrupt.shape); self.assertEqual(float(label), 1.0)
            self.assertEqual(returned_path, str(path)); self.assertEqual(len(dataset[0]), 4)

    def test_validation_only_promotion_selection(self) -> None:
        summaries = []
        for name, worst in (("baseline", 0.90), ("mild", 0.92), ("medium", 0.91), ("strong", 0.89)):
            row = {"candidate_id": name, "consistency_setting": name, "status": "succeeded",
                   "clean_validation_balanced_accuracy": 0.965,
                   "worst_transformed_validation_balanced_accuracy": worst,
                   "mean_transformed_validation_balanced_accuracy": worst + 0.02,
                   "inference_multiplier": 1, "total_deployment_parameter_count": 10}
            summaries.append({"selection_split": "validation", "final_test_evaluated": False, "results": [row]})
        selected = select_promotion_setting(summaries, 0.9681)
        self.assertEqual(selected["selected_setting"], "mild")
        bad = [{"selection_split": "test", "final_test_evaluated": True, "results": []}]
        with self.assertRaisesRegex(ValueError, "validation-only"): select_promotion_setting(bad, 0.9681)

    def test_four_bounded_configs_and_common_contract(self) -> None:
        self.assertEqual(set(CONSISTENCY_CONFIGS), {"baseline", "mild", "medium", "strong"})
        self.assertIn("val_logits.npz", REQUIRED_FILES); self.assertIn("best_model.pt", REQUIRED_FILES)


if __name__ == "__main__": unittest.main()
