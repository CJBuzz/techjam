import unittest
from collections import Counter

import torch

from aigc_detector.phase3.data import ManifestRecord
from aigc_detector.phase3.r4 import (
    POLICIES,
    quality_metadata,
    select_promotion_policy,
    select_training_records,
    source_leakage_diagnostic,
)


def record(index, label, source, generator=None, quality="q1"):
    return ManifestRecord(path=f"/input/{source}/{index}.jpg", label=label, split="train", source=source,
                          generator=generator, width=256, height=256,
                          metadata={"sharpness_bin": quality, "compression_proxy_bin": quality})


class R4Tests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for index in range(20):
            self.records.append(record(index, 0, "real_a" if index < 15 else "real_b", quality=f"q{index % 2}"))
            self.records.append(record(index, 1, "fake_a" if index < 15 else "fake_b",
                                       "g1" if index < 15 else "g2", quality=f"q{index % 2}"))

    def test_class_and_source_balancing(self):
        selected, _ = select_training_records(self.records, 20, 42, "source_balanced")
        self.assertEqual(Counter(row.label for row in selected), {0: 10, 1: 10})
        real_sources = Counter(row.source for row in selected if row.label == 0)
        self.assertLessEqual(max(real_sources.values()) - min(real_sources.values()), 1)

    def test_quality_matching_and_no_inference_features(self):
        selected, metadata = select_training_records(self.records, 20, 42, "source_quality_matched")
        real = Counter(tuple(quality_metadata(row).values()) for row in selected if row.label == 0)
        fake = Counter(tuple(quality_metadata(row).values()) for row in selected if row.label == 1)
        self.assertEqual(real, fake)
        self.assertFalse(metadata["quality_features_are_model_inputs"])
        self.assertNotIn("label", quality_metadata(selected[0])); self.assertNotIn("source", quality_metadata(selected[0]))

    def test_no_source_identifier_enters_model_tuple(self):
        # R4 uses R3 PairedDataset, whose contract is pixels, pixels, label, path only.
        from aigc_detector.phase3.r3 import PairedDataset
        self.assertNotIn("source", PairedDataset.__getitem__.__code__.co_names)
        self.assertNotIn("generator", PairedDataset.__getitem__.__code__.co_names)

    def test_equal_training_budget(self):
        sizes = {policy: len(select_training_records(self.records, 20, 42, policy)[0]) for policy in POLICIES}
        self.assertEqual(set(sizes.values()), {20})

    def test_validation_only_selection(self):
        summaries = []
        for policy, worst in zip(POLICIES, (0.90, 0.92, 0.91), strict=True):
            row = {"candidate_id": f"r4:{policy}", "bias_policy": policy, "status": "succeeded",
                   "clean_validation_balanced_accuracy": 0.965,
                   "worst_transformed_validation_balanced_accuracy": worst,
                   "mean_transformed_validation_balanced_accuracy": worst + 0.01,
                   "inference_multiplier": 1, "total_deployment_parameter_count": 10}
            summaries.append({"selection_split": "validation", "final_test_evaluated": False, "results": [row]})
        self.assertEqual(select_promotion_policy(summaries, 0.9681)["selected_policy"], "source_balanced")
        summaries[0]["selection_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation-only"):
            select_promotion_policy(summaries, 0.9681)

    def test_source_probe_is_diagnostic_only(self):
        validation = [ManifestRecord(path=str(i), label=i % 2, split="validation", source=f"s{i % 2}") for i in range(8)]
        result = source_leakage_diagnostic(validation, torch.tensor([[float(i % 2), 0.0] for i in range(8)]))
        self.assertFalse(result["used_as_detector_input"]); self.assertEqual(result["status"], "succeeded")


if __name__ == "__main__": unittest.main()
