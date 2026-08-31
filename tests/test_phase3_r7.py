import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aigc_detector.data import ROBUSTNESS_CONDITIONS
from aigc_detector.phase3.r7 import (
    Candidate,
    MINIMUM_ENSEMBLE_GAIN,
    _score,
    diversity_rows,
    normalize_weights,
    search,
)
from aigc_detector.phase3.r7_locked_test import validate_lock


def candidate(name, logits=None, parameters=100):
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    if logits is None: logits = np.tile(np.array([-2., -1., 1., 2.]), (len(ROBUSTNESS_CONDITIONS), 1))
    metadata = {"total_deployment_parameter_count": parameters, "inference_multiplier": 1,
                "model_backbone": "toy", "input_resolution": 256}
    reconstruction = [{"checkpoint": f"/kaggle/input/{name}/best_model.pt", "backbone": "dinov3_vitl16",
                       "resolution": 256, "detector_type": "global", "base_temperature": 1.0}]
    return Candidate(name, "R4", logits, labels, metadata, reconstruction)


class R7Tests(unittest.TestCase):
    def test_weight_normalization_and_component_limit(self):
        self.assertEqual(normalize_weights((2, 2)), (.5, .5))
        with self.assertRaisesRegex(ValueError, "one to three"): normalize_weights((1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"): normalize_weights((1, -1))

    def test_parameter_limit(self):
        with self.assertRaisesRegex(ValueError, "<2B"):
            _score((candidate("a", parameters=1_999_999_999), candidate("b", parameters=1)), (.5, .5))

    def test_endpoint_single_model_behavior(self):
        item = candidate("one"); row, logits = _score((item,), (1,))
        np.testing.assert_array_equal(logits, item.logits); self.assertEqual(row["component_count"], 1)

    def test_diversity_calculations(self):
        left = candidate("left")
        right_logits = left.logits.copy(); right_logits[:, 0] *= -1
        rows = diversity_rows([left, candidate("right", right_logits)])
        self.assertEqual(len(rows), len(ROBUSTNESS_CONDITIONS))
        self.assertIn("error_disagreement", rows[0]); self.assertTrue(any(row["important_condition"] for row in rows))

    def test_simplicity_threshold_keeps_single(self):
        left, right = candidate("left"), candidate("right")
        _, winner, _ = search([left, right], baseline_clean=1.0)
        self.assertEqual(winner["component_count"], 1)
        self.assertEqual(winner["simplicity_threshold"], MINIMUM_ENSEMBLE_GAIN)

    def test_lock_validation_is_validation_only(self):
        lock = {"contract_version": 1, "selection_split": "validation", "final_test_evaluated": False,
                "search_permitted": False, "components": [{"checkpoint": "/kaggle/input/a/model.pt",
                "ensemble_weight": 1., "total_temperature": 1.}], "ensemble_weights": [1.],
                "decision_threshold": .5, "total_deployment_parameter_count": 100}
        validate_lock(lock)
        lock["selection_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation-only"): validate_lock(lock)

    def test_locked_module_has_no_search_dependency(self):
        import aigc_detector.phase3.r7_locked_test as locked
        source = inspect.getsource(locked)
        self.assertNotIn("from .r7 import", source); self.assertNotIn("def search", source)


if __name__ == "__main__": unittest.main()
