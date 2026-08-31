import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aigc_detector.phase3.artifacts import REQUIRED_FILES, validate_completion, write_artifact_contract
from aigc_detector.phase3.config import load_config, require_validation_selection
from aigc_detector.phase3.job import torchrun_command
from aigc_detector.phase3.kaggle import kernel_metadata, package_source, source_included
from aigc_detector.phase3.ranking import rank_candidates
from aigc_detector.phase3.runtime import WallClockGuard, optimizer_step_due, resolve_distributed


def complete_metrics() -> dict:
    return {
        "model_backbone": "toy", "trainable_parameter_count": 2,
        "total_deployment_parameter_count": 3, "input_resolution": 32,
        "training_data_counts": {"class": {"real": 1, "ai": 1}, "source": {}, "generator": {}},
        "clean_validation_balanced_accuracy": 0.97,
        "mean_transformed_validation_balanced_accuracy": 0.92,
        "worst_transformed_validation_balanced_accuracy": 0.90, "worst_condition": "resize_x0.25",
        "resize_x0.25_balanced_accuracy": 0.90, "noise_sigma0.10_balanced_accuracy": 0.91,
        "blur_sigma2.0_balanced_accuracy": 0.92, "mean_transformed_roc_auc": 0.95,
        "worst_transformed_roc_auc": 0.93, "clean_false_positive_rate": 0.03,
        "mean_transformed_false_positive_rate": 0.06, "inference_multiplier": 1,
        "clean_constraint_pass": True, "selection_split": "validation", "final_test_evaluated": False,
    }


class MutableClock:
    def __init__(self) -> None: self.value = 0.0
    def __call__(self) -> float: return self.value


class Phase3Tests(unittest.TestCase):
    def test_config_parsing_and_offline_final_test_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"experiment": "r1", "backbone": "toy",
                                        "distributed": {"expected_gpus": 2}}))
            config = load_config(path)
            self.assertEqual(config.distributed.expected_gpus, 2)
            self.assertEqual(config.precision, "fp16")
            self.assertFalse(config.runtime_internet_required)
            path.write_text(json.dumps({"experiment": "r1", "backbone": "toy", "selection_split": "test"}))
            with self.assertRaisesRegex(ValueError, "test is forbidden"): load_config(path)
        with self.assertRaises(ValueError): require_validation_selection("test")

    def test_ddp_two_gpu_and_fallbacks(self) -> None:
        ddp = resolve_distributed(2, {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"})
        self.assertTrue(ddp.distributed); self.assertEqual(ddp.world_size, 2); self.assertEqual(str(ddp.device), "cuda:1")
        single = resolve_distributed(1, {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"})
        self.assertFalse(single.distributed); self.assertEqual(single.world_size, 1); self.assertEqual(str(single.device), "cuda:0")
        cpu = resolve_distributed(0, {})
        self.assertEqual(str(cpu.device), "cpu"); self.assertFalse(cpu.distributed)
        self.assertEqual(torchrun_command("entrypoint.py", "config.json", 2)[:3],
                         ["torchrun", "--standalone", "--nproc_per_node=2"])

    def test_wall_clock_and_accumulation(self) -> None:
        clock = MutableClock(); guard = WallClockGuard(10, reserve_minutes=1, clock=clock)
        clock.value = 8 * 60; self.assertFalse(guard.should_stop())
        saved = []
        clock.value = 9 * 60; self.assertTrue(guard.should_stop())
        self.assertTrue(guard.stop_after_safe_unit(saved.append)); self.assertEqual(saved[0]["reason"], "wall_clock_guard")
        self.assertFalse(optimizer_step_due(0, 2)); self.assertTrue(optimizer_step_due(1, 2))

    def test_artifact_contract_and_completion_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            write_artifact_contract(output, {"experiment": "r1"}, complete_metrics(),
                                    [{"condition": "clean", "balanced_accuracy": 1.0}],
                                    np.array([0.1, 0.9]), np.array([0, 1]), {"weight": torch.ones(1)},
                                    {"selection_split": "validation", "final_test_evaluated": False})
            self.assertEqual({path.name for path in output.iterdir()}, set(REQUIRED_FILES))
            self.assertEqual(validate_completion(output)["completed"]["status"], "completed")
            (output / "COMPLETED.json").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"): validate_completion(output)

    def test_clean_constraint_ranking_and_cost_tie_break(self) -> None:
        rows = [
            {"candidate_id": "fast", "clean_validation_balanced_accuracy": 0.965,
             "worst_transformed_validation_balanced_accuracy": 0.91,
             "mean_transformed_validation_balanced_accuracy": 0.94, "inference_multiplier": 1},
            {"candidate_id": "slow", "clean_validation_balanced_accuracy": 0.965,
             "worst_transformed_validation_balanced_accuracy": 0.91,
             "mean_transformed_validation_balanced_accuracy": 0.94, "inference_multiplier": 3},
            {"candidate_id": "unclean", "clean_validation_balanced_accuracy": 0.94,
             "worst_transformed_validation_balanced_accuracy": 0.99,
             "mean_transformed_validation_balanced_accuracy": 0.99},
        ]
        ranked = rank_candidates(rows, 0.9681)
        values = {row["candidate_id"]: row["validation_rank"] for row in ranked}
        self.assertEqual(values["fast"], 1); self.assertEqual(values["slow"], 2); self.assertIsNone(values["unclean"])

    def test_kaggle_metadata_is_private_gpu_offline(self) -> None:
        metadata = kernel_metadata("user/phase3-r1", "R1", "entrypoint.py", ["user/data"])
        self.assertTrue(metadata["enable_gpu"]); self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])

    def test_source_packaging_excludes_runtime_and_heavy_files(self) -> None:
        self.assertFalse(source_included(Path("artifacts/model.pt")))
        self.assertFalse(source_included(Path("a/__pycache__/x.pyc")))
        self.assertTrue(source_included(Path("aigc_detector/model.py")))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"; root.mkdir(); (root / "keep.py").write_text("pass\n")
            (root / "data").mkdir(); (root / "data/raw.jpg").write_bytes(b"x")
            (root / "model.pt").write_bytes(b"x")
            output = Path(tmp) / "package"; package_source(root, output)
            self.assertTrue((output / "keep.py").is_file())
            self.assertFalse((output / "data/raw.jpg").exists()); self.assertFalse((output / "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
