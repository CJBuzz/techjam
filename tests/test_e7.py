import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigc_detector.experiments.e7 import (
    MODEL_MODES,
    STABILITY_SCALES,
    SUPPORTED_BINS,
    DescriptorDataset,
    descriptor_cache_valid,
    eligibility_summary,
    extract_descriptor_tasks,
    radial_bin_indices,
    radial_fft_descriptor,
    rank_results,
    require_validation_selection,
    select_features,
    select_eligible_winner,
)
from aigc_detector.model import FrozenEncoders


class E7Tests(unittest.TestCase):
    def test_dimensions_finite_and_normalized_on_constant_image(self) -> None:
        image = Image.new("RGB", (32, 48), "gray")
        for bins in SUPPORTED_BINS:
            descriptor = radial_fft_descriptor(image, bins)
            self.assertEqual(descriptor.shape, (bins,))
            self.assertTrue(bool(torch.isfinite(descriptor).all()))
            self.assertGreaterEqual(float(descriptor.sum()), 0.0)
            self.assertLessEqual(float(descriptor.sum()), 1.000001)

    def test_radial_bins_cover_every_spectrum_pixel(self) -> None:
        indices = radial_bin_indices(31, 47, 32)
        counts = np.bincount(indices.ravel(), minlength=32)
        self.assertEqual(int(counts.sum()), 31 * 47)
        self.assertTrue(((indices >= 0) & (indices < 32)).all())

    def test_resize_descriptor_is_deterministic(self) -> None:
        image = Image.fromarray(np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3), "RGB")
        resized = image.resize((24, 16), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
        self.assertTrue(torch.equal(radial_fft_descriptor(resized), radial_fft_descriptor(resized)))

    def test_worker_count_does_not_change_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(2):
                path = Path(tmp) / f"{index}.png"
                Image.new("RGB", (24, 24), (index * 50, 20, 30)).save(path)
                paths.append(path)
            tasks = [(path, "clean", 0) for path in paths]
            serial = extract_descriptor_tasks(tasks, (16,), 42, 2, 0)[16]
            parallel = extract_descriptor_tasks(tasks, (16,), 42, 2, 2)[16]
            self.assertTrue(torch.equal(serial, parallel))

    def test_cache_resume_validation_and_fft_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            manifest = {"bins": 16, "train_originals": 2, "train_repeats": 3,
                        "validation_originals": 2, "validation_conditions": ["clean", "jpeg_q90"],
                        "stability_originals": 2, "stability_scales": list(STABILITY_SCALES)}
            torch.save({"train": torch.zeros(6, 16), "validation": torch.zeros(4, 16),
                        "stability": torch.zeros(8, 16), "manifest": manifest}, path)
            self.assertTrue(descriptor_cache_valid(path, manifest))
            incompatible = {**manifest, "bins": 32}
            self.assertFalse(descriptor_cache_valid(path, incompatible))
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_selection("test")
        self.assertNotIn("radial", inspect.getsource(FrozenEncoders._fft_tensor).lower())

    def test_all_model_modes_have_expected_dimensions(self) -> None:
        fused = torch.zeros(2, 3072)
        radial = torch.zeros(2, 32)
        expected = {"radial_only": 32, "clip_radial": 544, "fused_radial": 3104}
        self.assertEqual(set(MODEL_MODES), set(expected))
        for mode, width in expected.items():
            self.assertEqual(select_features(fused, radial, mode).shape, (2, width))

    def test_eligible_candidate_is_ranked_and_selected(self) -> None:
        rows = [
            {"model_mode": "radial_only", "radial_bins": 16, "status": "succeeded",
             "clean_validation_balanced_accuracy": 0.96,
             "mean_transformed_validation_balanced_accuracy": 0.80,
             "worst_transformed_validation_balanced_accuracy": 0.70},
            {"model_mode": "fused_radial", "radial_bins": 16, "status": "succeeded",
             "clean_validation_balanced_accuracy": 0.97,
             "mean_transformed_validation_balanced_accuracy": 0.90,
             "worst_transformed_validation_balanced_accuracy": 0.85},
        ]
        ranked = rank_results(rows, baseline_clean=0.97)
        winner = select_eligible_winner(ranked)
        self.assertEqual(winner["model_mode"], "fused_radial")
        self.assertEqual(winner["rank"], 1)

    def test_all_clean_constraint_failures_are_valid_negative_result(self) -> None:
        rows = [{"model_mode": "radial_only", "radial_bins": 16, "status": "succeeded",
                 "clean_validation_balanced_accuracy": 0.90,
                 "mean_transformed_validation_balanced_accuracy": 0.85,
                 "worst_transformed_validation_balanced_accuracy": 0.80}]
        ranked = rank_results(rows, baseline_clean=0.97)
        summary = eligibility_summary(ranked, baseline_clean=0.97)
        self.assertIsNone(select_eligible_winner(ranked))
        self.assertIsNone(summary["eligible_winner"])
        self.assertTrue(summary["no_eligible_candidate"])
        self.assertIn("No succeeded E7 candidate", summary["reason"])

    def test_failed_rows_do_not_trigger_empty_minimum(self) -> None:
        rows = [{"model_mode": "clip_radial", "radial_bins": 64, "status": "failed",
                 "failure_reason": "Input contains NaN"}]
        ranked = rank_results(rows, baseline_clean=0.97)
        self.assertIsNone(select_eligible_winner(ranked))
        self.assertTrue(eligibility_summary(ranked, 0.97)["no_eligible_candidate"])


if __name__ == "__main__":
    unittest.main()
