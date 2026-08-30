import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from aigc_detector.e4b import AdaptiveFusionHead
from aigc_detector.e4c import (
    INTERVENTION_MODES,
    INTERVENTION_WEIGHTS,
    add_deltas,
    intervention_logits,
    validate_interventions,
    write_outputs,
)


class E4cTests(unittest.TestCase):
    def test_forced_weights_are_explicit_and_normalized(self) -> None:
        validate_interventions()
        self.assertEqual(INTERVENTION_MODES, (
            "learned", "equal", "global_mean", "semantic_heavy", "forensic_heavy",
            "laplacian_heavy", "fft_heavy",
        ))
        for weights in INTERVENTION_WEIGHTS.values():
            self.assertAlmostEqual(sum(weights), 1.0)

    def test_learned_preserves_model_and_forced_gate_bypasses_it(self) -> None:
        model = AdaptiveFusionHead("adaptive").eval()
        features = torch.randn(4, 3072)
        learned, learned_weights = intervention_logits(model, features, "learned")
        self.assertTrue(torch.allclose(learned, model(features)))
        equal, equal_weights = intervention_logits(model, features, "equal")
        expected = model.modality_logits(features).mean(1)
        self.assertTrue(torch.allclose(equal, expected))
        self.assertTrue(torch.allclose(equal_weights, torch.full((4, 3), 1 / 3)))
        self.assertNotEqual(learned_weights.data_ptr(), equal_weights.data_ptr())

    def test_no_condition_specific_oracle_input(self) -> None:
        self.assertNotIn("condition", inspect.signature(intervention_logits).parameters)
        self.assertTrue(all(not isinstance(value, dict) for value in INTERVENTION_WEIGHTS.values()))

    def test_output_table_structure(self) -> None:
        template = {
            "weights_or_mode": [1 / 3] * 3, "deployment_capable": True,
            "clean_validation_balanced_accuracy": 0.8,
            "mean_transformed_validation_balanced_accuracy": 0.7,
            "worst_transformed_validation_balanced_accuracy": 0.6,
            "worst_condition": "resize_x0.25", "per_condition_balanced_accuracy": {"clean": 0.8},
            "mean_transformed_false_positive_rate": 0.2, "temperature_and_threshold_refit": False,
        }
        rows = add_deltas([
            {**template, "intervention": "learned"},
            {**template, "intervention": "equal"},
            {**template, "intervention": "global_mean"},
        ])
        shifts = [{"condition": "clean", "semantic_weight": 1 / 3, "laplacian_weight": 1 / 3,
                   "fft_weight": 1 / 3, "l1_distance_from_clean": 0.0,
                   "semantic_weight_change": 0.0, "laplacian_weight_change": 0.0,
                   "fft_weight_change": 0.0}]
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(rows, shifts, Path(tmp))
            self.assertTrue((Path(tmp) / "intervention_summary.csv").is_file())
            self.assertTrue((Path(tmp) / "condition_gate_shift.csv").is_file())


if __name__ == "__main__":
    unittest.main()
