import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from aigc_detector.data import ROBUSTNESS_CONDITIONS
from aigc_detector.phase3.data import ManifestRecord
from aigc_detector.phase3.r1 import VisionDetector
from aigc_detector.phase3.r5 import (
    EXPERTS,
    PARAMETER_LIMIT,
    SpecialistPairedDataset,
    assert_checkpoint_compatible,
    blend_logits,
    deployment_parameters,
    ensemble_candidates,
    specialist_chain,
)


class DummyProcessor:
    def __call__(self, images, size, return_tensors):
        array = np.asarray(images.resize((size["width"], size["height"])), dtype=np.float32) / 255
        return {"pixel_values": torch.tensor(array).permute(2, 0, 1).unsqueeze(0)}


class InterpolatingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.projection = torch.nn.Linear(3, 4); self.interpolated = False

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        self.interpolated = interpolate_pos_encoding
        pooled = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(pooler_output=self.projection(pooled))


class R5Tests(unittest.TestCase):
    def test_resolution_specific_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.png"; Image.new("RGB", (20, 15), "blue").save(path)
            row = ManifestRecord(path=str(path), label=1, split="train", unique_id="one")
            for expert, expected in (("low", 256), ("high", 384)):
                dataset = SpecialistPairedDataset([row], DummyProcessor(), expected, 42, expert, 2)
                clean, corrupt, label, _ = dataset[0]
                self.assertEqual(clean.shape, (3, expected, expected)); self.assertEqual(clean.shape, corrupt.shape)
                self.assertEqual(float(label), 1.0)

    def test_profiles_are_deterministic_and_distinct(self):
        self.assertEqual(specialist_chain("low", 42, "x", 0), specialist_chain("low", 42, "x", 0))
        low = [specialist_chain("low", 42, str(i), 0)[0] for i in range(20)]
        self.assertTrue(any(item.startswith(("resize", "blur", "noise")) for item in low))
        self.assertEqual(EXPERTS["high"]["resolution"], 384)

    def test_checkpoint_compatibility(self):
        model = torch.nn.Linear(3, 1); state = {"state_dict": model.state_dict()}
        assert_checkpoint_compatible(model, state)
        bad = {"state_dict": {"weight": torch.zeros(2, 3), "bias": torch.zeros(1)}}
        with self.assertRaisesRegex(ValueError, "shape_mismatch"):
            assert_checkpoint_compatible(model, bad)

    def test_supported_positional_interpolation(self):
        backbone = InterpolatingBackbone(); model = VisionDetector(backbone, 4)
        self.assertEqual(model(torch.randn(2, 3, 384, 384)).shape, (2,))
        self.assertTrue(backbone.interpolated)

    def test_alpha_endpoints_reproduce_experts(self):
        low = np.array([[1.0, 2.0]]); high = np.array([[-2.0, 3.0]])
        np.testing.assert_array_equal(blend_logits(low, high, 1.0), low)
        np.testing.assert_array_equal(blend_logits(low, high, 0.0), high)

    def test_parameter_count_and_limit(self):
        self.assertEqual(deployment_parameters(500, 700), 1200)
        with self.assertRaisesRegex(ValueError, "<2B"):
            deployment_parameters(PARAMETER_LIMIT - 1, 1)

    def test_validation_only_ensemble_selection(self):
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        low = np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), (len(ROBUSTNESS_CONDITIONS), 1))
        high = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), (len(ROBUSTNESS_CONDITIONS), 1))
        base = {"status": "succeeded", "model_backbone": "toy", "trainable_parameter_count": 5,
                "total_deployment_parameter_count": 100, "input_resolution": 256,
                "training_data_counts": {}, "clean_validation_balanced_accuracy": 1.0,
                "mean_transformed_validation_balanced_accuracy": 1.0,
                "worst_transformed_validation_balanced_accuracy": 1.0,
                "inference_multiplier": 1.0, "selection_split": "validation", "final_test_evaluated": False}
        rows, _ = ensemble_candidates(base, {**base, "input_resolution": 384, "inference_multiplier": 2.25},
                                      low, high, labels, 1.0)
        self.assertTrue(rows); self.assertTrue(all(row["selection_split"] == "validation" for row in rows))
        self.assertTrue(all(row["final_test_evaluated"] is False for row in rows))
        self.assertEqual(sum(row["validation_rank"] == 1 for row in rows), 1)


if __name__ == "__main__": unittest.main()
