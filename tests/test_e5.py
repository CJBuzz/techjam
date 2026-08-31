import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from aigc_detector.experiments.e5 import (
    QUALITY_FEATURE_NAMES,
    fit_binned_thresholds,
    fit_continuous_threshold,
    global_config,
    load_locked,
    quality_vector,
    require_validation_fit,
    thresholds_for,
    validate_quality_schema,
)


class E5Tests(unittest.TestCase):
    def test_quality_schema_is_safe_and_deterministic(self) -> None:
        validate_quality_schema()
        image = Image.new("RGB", (16, 16), "gray")
        logits = torch.tensor([0.1, 0.2, 0.3])
        self.assertTrue(torch.equal(quality_vector(image, logits), quality_vector(image, logits)))
        self.assertEqual(len(QUALITY_FEATURE_NAMES), 7)
        self.assertNotIn("condition", inspect.signature(quality_vector).parameters)

    def test_global_threshold_reproduces_baseline(self) -> None:
        quality = torch.randn(3, 7)
        self.assertTrue(torch.equal(thresholds_for(quality, global_config(0.42)), torch.full((3,), 0.42)))

    def test_binned_boundaries_are_validation_only(self) -> None:
        probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9])
        quality = torch.arange(28.0).view(4, 7)
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        config = fit_binned_thresholds(probabilities, quality, labels, 0.5, bins=2)
        self.assertEqual(config["fit_split"], "validation")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            fit_binned_thresholds(probabilities, quality, labels, 0.5, bins=2, split="test")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_fit("test")

    def test_continuous_threshold_is_bounded(self) -> None:
        probabilities = torch.linspace(0.05, 0.95, 20)
        quality = torch.randn(20, 7, generator=torch.Generator().manual_seed(42))
        labels = (probabilities > 0.5).float()
        config = fit_continuous_threshold(probabilities, quality, labels, 0.5, max_delta=0.08)
        thresholds = thresholds_for(quality * 100, config)
        self.assertTrue(bool(((thresholds >= 0.42) & (thresholds <= 0.58)).all()))
        self.assertEqual(config["parameter_count"], 8)
        self.assertNotIn("condition", inspect.signature(fit_continuous_threshold).parameters)

    def test_locked_config_round_trip(self) -> None:
        payload = {"selection_split": "validation", "test_rows_used_for_selection": False,
                   "quality_features": list(QUALITY_FEATURE_NAMES), "calibration": global_config(0.5)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "locked.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_locked(path), payload)


if __name__ == "__main__":
    unittest.main()
