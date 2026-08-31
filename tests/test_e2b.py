import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigc_detector.experiments.e2b import (
    AGGREGATIONS,
    ATOMIC_VIEWS,
    POLICIES,
    aggregate_logits,
    atomic_view,
    construct_policy_views,
    load_locked_policy,
    load_or_build_logit_cache,
    rank_policy_results,
    require_validation_search,
)


class E2bTests(unittest.TestCase):
    def test_curated_policy_views_are_exact_and_deterministic(self) -> None:
        image = Image.fromarray(np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3), "RGB")
        self.assertEqual(ATOMIC_VIEWS, ("identity", "jpeg90", "resize0.75", "resize0.5", "blur0.5"))
        self.assertEqual(len(POLICIES), 12)
        self.assertIn(("identity", "jpeg90", "resize0.5"), POLICIES.values())
        for name in ATOMIC_VIEWS:
            first = atomic_view(image, name, 42, "sample")
            second = atomic_view(image, name, 42, "sample")
            self.assertEqual(first.size, image.size)
            self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        self.assertEqual(len(construct_policy_views(image, "identity+jpeg90+resize0.5", 42, "x")), 3)

    def test_mean_and_median_logit_aggregation(self) -> None:
        logits = torch.tensor([[0.0, 2.0, 10.0], [-3.0, 1.0, 2.0]])
        self.assertTrue(torch.equal(aggregate_logits(logits, "mean"), logits.mean(1)))
        self.assertTrue(torch.equal(aggregate_logits(logits, "median"), torch.tensor([2.0, 1.0])))
        self.assertEqual(float(aggregate_logits(torch.tensor([[0.0, 2.0]]), "median")), 1.0)
        self.assertEqual(AGGREGATIONS, ("mean", "median"))

    def test_atomic_logit_cache_reuses_completed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            calls = []
            manifest = {"split": "validation"}
            def builder():
                calls.append(1)
                return {"logits": torch.ones(1), "labels": torch.zeros(1)}
            load_or_build_logit_cache(path, manifest, builder)
            load_or_build_logit_cache(path, manifest, builder)
            self.assertEqual(len(calls), 1)

    def test_validation_ranking_and_efficiency_tie_break(self) -> None:
        base = {
            "aggregation": "mean", "selection_split": "validation", "test_rows_used_for_selection": False,
            "clean_validation_balanced_accuracy": 0.90,
            "mean_transformed_validation_balanced_accuracy": 0.80,
            "worst_transformed_validation_balanced_accuracy": 0.70,
        }
        rows = [
            {**base, "policy_name": "identity", "number_of_views": 1},
            {**base, "policy_name": "identity+jpeg90", "number_of_views": 2},
            {**base, "policy_name": "identity+jpeg90+resize0.5", "number_of_views": 3},
        ]
        ranked = rank_policy_results(rows)
        self.assertEqual(next(row for row in ranked if row["rank"] == 1)["policy_name"], "identity")
        with self.assertRaisesRegex(ValueError, "validation-only"):
            require_validation_search("test")

    def test_locked_test_accepts_only_one_validation_winner(self) -> None:
        locked = {
            "schema_version": 1, "selection_split": "validation",
            "test_rows_used_for_selection": False, "checkpoint": "/tmp/model.pt",
            "policy_name": "identity+jpeg90", "atomic_views": ["identity", "jpeg90"],
            "aggregation": "median", "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "winner.json"
            path.write_text(json.dumps(locked), encoding="utf-8")
            self.assertEqual(load_locked_policy(path), locked)
            locked["alternative_policies"] = ["identity"]
            path.write_text(json.dumps(locked), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one winner"):
                load_locked_policy(path)


if __name__ == "__main__":
    unittest.main()
