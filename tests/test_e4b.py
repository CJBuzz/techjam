import tempfile
import unittest
from pathlib import Path

import torch

from aigc_detector.e4b import (
    AdaptiveFusionHead,
    GATE_MODES,
    QUALITY_FEATURE_NAMES,
    aggregate_gate_statistics,
    load_adaptive_checkpoint,
    parameter_count,
    quality_vector,
    require_validation_selection,
    save_adaptive_checkpoint,
    split_modalities,
)


class E4bTests(unittest.TestCase):
    def test_feature_blocks_quality_determinism_and_no_leakage(self) -> None:
        features = torch.randn(4, 3072, generator=torch.Generator().manual_seed(42))
        blocks = split_modalities(features)
        self.assertEqual([block.shape[1] for block in blocks], [512, 1280, 1280])
        logits = torch.randn(4, 3, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(quality_vector(features, logits), quality_vector(features, logits)))
        forbidden = ("label", "target", "source", "generator", "condition", "filename", "path")
        self.assertFalse(any(token in name for name in QUALITY_FEATURE_NAMES for token in forbidden))

    def test_all_modes_forward_weights_and_parameter_budget(self) -> None:
        features = torch.randn(5, 3072)
        expected_counts = {"fixed": 3075, "global": 3078, "adaptive": 3686}
        self.assertEqual(GATE_MODES, ("fixed", "global", "adaptive"))
        for mode in GATE_MODES:
            model = AdaptiveFusionHead(mode)
            logits, weights = model(features, return_weights=True)
            self.assertEqual(logits.shape, (5,))
            self.assertEqual(weights.shape, (5, 3))
            self.assertTrue(torch.allclose(weights.sum(1), torch.ones(5)))
            self.assertEqual(parameter_count(model), expected_counts[mode])
            self.assertLess(parameter_count(model), 100_000)
        adaptive = AdaptiveFusionHead("adaptive")
        self.assertTrue(torch.allclose(adaptive.gate_weights(features), torch.full((5, 3), 1 / 3)))

    def test_checkpoint_round_trip(self) -> None:
        model = AdaptiveFusionHead("adaptive")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_adaptive_checkpoint(path, model, 1.2, 0.45, {"selection_split": "validation"})
            loaded, temperature, threshold, metadata = load_adaptive_checkpoint(path, torch.device("cpu"))
            self.assertFalse(loaded.training)
            self.assertEqual((temperature, threshold), (1.2, 0.45))
            self.assertEqual(metadata["selection_split"], "validation")

    def test_validation_only_and_gate_statistics(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_selection("test")
        weights = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        labels = torch.tensor([0.0, 1.0, 1.0])
        stats = aggregate_gate_statistics(weights, labels, ["clean", "clean", "resize_x0.25"])
        self.assertEqual(stats["clean:overall"]["count"], 2)
        self.assertTrue(torch.allclose(
            torch.tensor(stats["resize_x0.25:fake"]["weights"]), torch.tensor([0.4, 0.4, 0.2])
        ))


if __name__ == "__main__":
    unittest.main()
