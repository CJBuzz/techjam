import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from aigc_detector.phase2 import DEPLOYABLE_SOURCES, DIAGNOSTIC_SOURCES, aggregate, rank_candidates


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_track5_phase2.sh"
LOCKED = ROOT / "scripts/evaluate_locked_phase2_candidate.sh"


def candidate(name: str, worst: float, mean: float, *, external: float = 0.0,
              constraint: bool = True, multiplier: float = 1.0) -> dict:
    return {
        "experiment": name, "variant": name, "experiment_type": "deployment_candidate",
        "checkpoint_config_artifact": "model.pt", "clean_validation_balanced_accuracy": 0.95,
        "mean_transformed_validation_balanced_accuracy": mean,
        "worst_transformed_validation_balanced_accuracy": worst, "worst_condition": "resize_x0.25",
        "resize_x0.25_balanced_accuracy": worst, "resize_x0.5_balanced_accuracy": mean,
        "blur_sigma2.0_balanced_accuracy": None, "noise_sigma0.10_balanced_accuracy": None,
        "mean_transformed_roc_auc": None, "worst_transformed_roc_auc": None,
        "clean_false_positive_rate": None, "mean_transformed_false_positive_rate": None,
        "trainable_parameter_count": 10, "inference_multiplier": multiplier,
        "external_roc_auc": external, "external_recall": None,
        "clean_constraint_pass": constraint, "status": "succeeded", "validation_rank": None,
    }


class Phase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.runner_tmp.cleanup)

    def run_runner(self, env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(RUNNER)], cwd=ROOT, text=True, capture_output=True,
                              env={**os.environ, "TRACK5_ROOT": str(Path(self.runner_tmp.name) / "track5"), **env},
                              check=False)

    def test_start_end_and_required_failure_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "done.json"
            artifact.write_text(json.dumps({"selection_split": "validation", "results": [{}]}))
            marker = Path(tmp) / "ran"
            result = self.run_runner({"START_STAGE": "3", "END_STAGE": "3", "FORCE_STAGE": "3",
                                      "PHASE2_STAGE_3_ARTIFACT": str(artifact),
                                      "PHASE2_STAGE_3_COMMAND": f"touch '{marker}'"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(marker.exists())
            self.assertNotIn("STAGE 2 START", result.stdout)
            self.assertNotIn("STAGE 4 START", result.stdout)

            failed = self.run_runner({"START_STAGE": "3", "END_STAGE": "4", "FORCE_STAGE": "3",
                                      "PHASE2_STAGE_3_ARTIFACT": str(artifact),
                                      "PHASE2_STAGE_3_COMMAND": "exit 7",
                                      "PHASE2_STAGE_4_ARTIFACT": str(artifact),
                                      "PHASE2_STAGE_4_COMMAND": "true"})
            self.assertEqual(failed.returncode, 7)
            self.assertIn("STAGE 3 FAILED", failed.stdout)
            self.assertNotIn("STAGE 3 COMPLETE", failed.stdout)
            self.assertNotIn("STAGE 4 START", failed.stdout)

    def test_completion_is_skipped_only_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.json"
            valid.write_text(json.dumps({"selection_split": "validation", "results": [{}]}))
            skipped = self.run_runner({"START_STAGE": "2", "END_STAGE": "2",
                                       "PHASE2_STAGE_2_ARTIFACT": str(valid),
                                       "PHASE2_STAGE_2_COMMAND": "exit 9"})
            self.assertEqual(skipped.returncode, 0)
            self.assertIn("STAGE 2 SKIPPED", skipped.stdout)
            invalid = Path(tmp) / "invalid.json"; invalid.write_text("{}")
            rerun = self.run_runner({"START_STAGE": "2", "END_STAGE": "2",
                                     "PHASE2_STAGE_2_ARTIFACT": str(invalid),
                                     "PHASE2_STAGE_2_COMMAND": "exit 9"})
            self.assertEqual(rerun.returncode, 9)

    def test_ranking_uses_validation_robustness_not_external_metrics(self) -> None:
        rows = [candidate("robust", 0.90, 0.91, external=0.1),
                candidate("external", 0.89, 0.99, external=1.0),
                candidate("ineligible", 0.99, 0.99, constraint=False)]
        ranked = rank_candidates(rows)
        ranks = {row["experiment"]: row["validation_rank"] for row in ranked}
        self.assertEqual(ranks["robust"], 1)
        self.assertEqual(ranks["external"], 2)
        self.assertIsNone(ranks["ineligible"])

    def test_aggregate_excludes_diagnostics_and_locks_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_row = {
                "status": "succeeded", "clean_constraint_pass": True,
                "selection_split": "validation",
                "clean_validation_balanced_accuracy": 0.95,
                "mean_transformed_validation_balanced_accuracy": 0.90,
                "worst_transformed_validation_balanced_accuracy": 0.85,
                "worst_condition": "resize_x0.25",
            }
            for index, (experiment, relative) in enumerate(DEPLOYABLE_SOURCES):
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                row = {**base_row, "worst_transformed_validation_balanced_accuracy": 0.85 + index / 1000,
                       "mode": "adaptive" if experiment.startswith("E4b") else experiment}
                document = row if experiment.startswith("E4b") else {"selection_split": "validation", "results": [row]}
                path.write_text(json.dumps(document))
            for experiment, relative in DIAGNOSTIC_SOURCES:
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"selection_split": "validation", "results": [{"name": experiment}]}))
            recommendation = aggregate(root)
            self.assertEqual(recommendation["selection_split"], "validation")
            self.assertFalse(recommendation["final_test_evaluated"])
            summary = json.loads((root / "phase2/phase2_summary.json").read_text())
            self.assertTrue(all(row["experiment_type"] == "deployment_candidate" for row in summary["results"]))
            diagnostics = json.loads((root / "phase2/diagnostic_summary.json").read_text())
            self.assertTrue(diagnostics["excluded_from_deployment_ranking"])

    def test_runner_never_streams_or_runs_final_test_and_locked_script_never_searches(self) -> None:
        runner = RUNNER.read_text()
        locked = LOCKED.read_text()
        self.assertNotIn("streaming_cache", runner)
        self.assertNotIn("--split test", runner)
        self.assertNotIn(" locked-test ", runner)
        self.assertNotIn(" search ", locked)
        self.assertNotIn(" sweep ", locked)


if __name__ == "__main__":
    unittest.main()
