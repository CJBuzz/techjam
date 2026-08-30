import json
import tempfile
import unittest
from pathlib import Path

import torch

from aigc_detector.data import ROBUSTNESS_CONDITIONS
from aigc_detector.phase3.artifacts import REQUIRED_FILES
from aigc_detector.phase3.data import ManifestRecord
from aigc_detector.phase3.r2 import (
    CURRICULUM_PROBABILITIES,
    balanced_training_records,
    curriculum_chain,
    curriculum_level,
    load_r1_candidate,
    select_promotion_regime,
    validate_no_split_leakage,
)


def record(index: int, label: int, source: str, generator: str | None = None,
           split: str = "train", identity: str | None = None) -> ManifestRecord:
    return ManifestRecord(path=f"/{source}/{index}.png", label=label, split=split, source=source,
                          generator=generator, width=32, height=32, unique_id=identity or f"id-{index}-{split}")


class R2Tests(unittest.TestCase):
    def test_source_balanced_sampler_and_exact_class_balance(self) -> None:
        rows = []
        for index in range(30): rows.append(record(index, 0, "real_a" if index < 25 else "real_b"))
        for index in range(30, 60): rows.append(record(index, 1, "fake", f"g{index % 3}"))
        selected, stats = balanced_training_records(rows, 40, 42, max_fake_per_generator=8)
        labels = [row.label for row in selected]
        self.assertEqual(labels.count(0), labels.count(1)); self.assertEqual(len(selected), 40)
        self.assertGreater(stats["source"]["real_b"], 0)
        generators = {}
        for row in selected:
            if row.label: generators[row.generator] = generators.get(row.generator, 0) + 1
        self.assertLessEqual(max(generators.values()), 8)

    def test_curriculum_probabilities_and_compositions(self) -> None:
        self.assertEqual(curriculum_level(0.0), "early"); self.assertEqual(curriculum_level(0.5), "middle")
        self.assertEqual(curriculum_level(1.0), "late")
        for probabilities in CURRICULUM_PROBABILITIES.values():
            self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        early = {curriculum_chain("compound_curriculum", 0.0, 42, str(index)) for index in range(100)}
        late = {curriculum_chain("compound_curriculum", 1.0, 42, str(index)) for index in range(100)}
        self.assertTrue(all(len(chain) <= 1 for chain in early))
        self.assertTrue(any(len(chain) >= 2 for chain in late))
        self.assertEqual(curriculum_chain("compound_curriculum", 1.0, 42, "same"),
                         curriculum_chain("compound_curriculum", 1.0, 42, "same"))

    def test_exact_evaluation_conditions_remain_atomic(self) -> None:
        self.assertTrue(all("+" not in condition for condition in ROBUSTNESS_CONDITIONS))
        self.assertIn("resize_x0.25", ROBUSTNESS_CONDITIONS)

    def test_split_leakage_is_rejected(self) -> None:
        rows = [record(1, 0, "a", split="train", identity="duplicate"),
                record(2, 0, "a", split="validation", identity="duplicate")]
        with self.assertRaisesRegex(ValueError, "crosses train/validation"): validate_no_split_leakage(rows)

    def test_r1_recommendation_resolves_backbone_agnostic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / "candidates/last2/best_model.pt"
            checkpoint.parent.mkdir(parents=True); torch.save({}, checkpoint)
            recommendation = root / "recommended_candidate.json"
            recommendation.write_text(json.dumps({"selection_split": "validation", "final_test_evaluated": False,
                "candidate": {"model_backbone": "siglip2_large_256", "training_mode": "last2",
                              "clean_constraint_pass": True,
                              "checkpoint_relative_path": "candidates/last2/best_model.pt"}}))
            candidate, resolved = load_r1_candidate(recommendation, root)
            self.assertEqual(candidate["model_backbone"], "siglip2_large_256"); self.assertEqual(resolved, checkpoint)

    def test_compound_is_not_promoted_when_validation_is_worse(self) -> None:
        def winner(regime, worst):
            return {"candidate_id": regime, "regime": regime, "status": "succeeded",
                    "clean_validation_balanced_accuracy": 0.965,
                    "worst_transformed_validation_balanced_accuracy": worst,
                    "mean_transformed_validation_balanced_accuracy": worst + 0.02,
                    "total_deployment_parameter_count": 10, "inference_multiplier": 1}
        single = {"selection_split": "validation", "final_test_evaluated": False,
                  "eligible_winner": winner("single_transform", 0.92)}
        compound = {"selection_split": "validation", "final_test_evaluated": False,
                    "eligible_winner": winner("compound_curriculum", 0.90)}
        selected = select_promotion_regime(single, compound, 0.9681)
        self.assertEqual(selected["selected_regime"], "single_transform")
        self.assertEqual(selected["selection_split"], "validation")

    def test_common_artifact_contract_preserves_validation_logits(self) -> None:
        self.assertIn("val_logits.npz", REQUIRED_FILES)
        self.assertIn("COMPLETED.json", REQUIRED_FILES)


if __name__ == "__main__": unittest.main()
