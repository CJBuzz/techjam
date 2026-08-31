import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from aigc_detector.phase3.artifacts import REQUIRED_FILES
from aigc_detector.phase3.r1 import (
    VisionDetector,
    configure_trainable_layers,
    memory_preflight,
    validate_offline_asset_path,
)
from aigc_detector.phase3.ranking import rank_candidates


class FakeBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__(); self.linear = nn.Linear(width, width); self.norm = nn.LayerNorm(width)

    def forward(self, values): return self.norm(self.linear(values))


class FakeBackbone(nn.Module):
    def __init__(self, width: int = 8, blocks: int = 6) -> None:
        super().__init__(); self.encoder = nn.Module(); self.encoder.layers = nn.ModuleList(
            [FakeBlock(width) for _ in range(blocks)]
        ); self.final_norm = nn.LayerNorm(width)

    def forward(self, pixel_values):
        values = pixel_values
        for block in self.encoder.layers: values = block(values)
        return SimpleNamespace(last_hidden_state=self.final_norm(values))


class R1Tests(unittest.TestCase):
    def test_adapter_output_shape(self) -> None:
        detector = VisionDetector(FakeBackbone(), 8)
        self.assertEqual(detector(torch.randn(3, 4, 8)).shape, (3,))

    def test_linear_mode_really_freezes_backbone(self) -> None:
        detector = VisionDetector(FakeBackbone(), 8)
        counts = configure_trainable_layers(detector.backbone, detector.classifier, "linear_head")
        self.assertTrue(all(not parameter.requires_grad for parameter in detector.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in detector.classifier.parameters()))
        self.assertEqual(counts["trainable_parameter_count"], 9)

    def test_last2_and_last4_select_final_blocks_and_norms(self) -> None:
        for mode, count in (("last2", 2), ("last4", 4)):
            detector = VisionDetector(FakeBackbone(), 8)
            configure_trainable_layers(detector.backbone, detector.classifier, mode)
            blocks = detector.backbone.encoder.layers
            for block in blocks[:-count]:
                self.assertFalse(block.linear.weight.requires_grad)
            for block in blocks[-count:]:
                self.assertTrue(block.linear.weight.requires_grad)
            self.assertTrue(detector.backbone.final_norm.weight.requires_grad)

    def test_offline_asset_loading_and_optional_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "asset"; asset.mkdir(); (asset / "config.json").write_text("{}")
            self.assertEqual(validate_offline_asset_path(asset), asset)
            self.assertIsNone(validate_offline_asset_path(Path(tmp) / "missing", optional=True))
            with self.assertRaises(FileNotFoundError): validate_offline_asset_path(Path(tmp) / "missing")

    def test_optional_memory_preflight_skips_unsafe_so400m(self) -> None:
        safe, reason = memory_preflight(15.0, optional=True, cuda_totals=[14.5, 14.5])
        self.assertFalse(safe); self.assertIn("below required", reason)
        with self.assertRaises(RuntimeError): memory_preflight(15.0, optional=False, cuda_totals=[14.5])

    def test_validation_only_ranking_prefers_cheaper_within_point_two_pp(self) -> None:
        rows = [
            {"candidate_id": "cheap", "status": "succeeded", "clean_validation_balanced_accuracy": 0.965,
             "worst_transformed_validation_balanced_accuracy": 0.9150,
             "mean_transformed_validation_balanced_accuracy": 0.945, "inference_multiplier": 1,
             "total_deployment_parameter_count": 100},
            {"candidate_id": "large", "status": "succeeded", "clean_validation_balanced_accuracy": 0.965,
             "worst_transformed_validation_balanced_accuracy": 0.9159,
             "mean_transformed_validation_balanced_accuracy": 0.945, "inference_multiplier": 1,
             "total_deployment_parameter_count": 200},
        ]
        ranked = rank_candidates(rows, 0.9681, effective_tie=0.002)
        self.assertEqual(next(row for row in ranked if row["validation_rank"] == 1)["candidate_id"], "cheap")
        self.assertTrue(all(row["selection_split"] == "validation" for row in ranked))

    def test_common_contract_includes_future_r7_logits(self) -> None:
        self.assertIn("val_logits.npz", REQUIRED_FILES)
        self.assertIn("best_model.pt", REQUIRED_FILES)


if __name__ == "__main__":
    unittest.main()
