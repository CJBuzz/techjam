from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


REQUIRED_FILES = (
    "run_config.json", "provenance.json", "metrics.json", "per_condition.csv",
    "val_logits.npz", "candidate.json", "best_model.pt", "COMPLETED.json",
)

REQUIRED_METRICS = {
    "model_backbone", "trainable_parameter_count", "total_deployment_parameter_count",
    "input_resolution", "training_data_counts", "clean_validation_balanced_accuracy",
    "mean_transformed_validation_balanced_accuracy", "worst_transformed_validation_balanced_accuracy",
    "worst_condition", "resize_x0.25_balanced_accuracy", "noise_sigma0.10_balanced_accuracy",
    "blur_sigma2.0_balanced_accuracy", "mean_transformed_roc_auc", "worst_transformed_roc_auc",
    "clean_false_positive_rate", "mean_transformed_false_positive_rate", "inference_multiplier",
    "clean_constraint_pass", "selection_split", "final_test_evaluated",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def provenance() -> dict[str, Any]:
    git_commit = None
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "platform": platform.platform(), "git_commit": git_commit, "runtime_internet_used": False}


def write_artifact_contract(output: Path, config: dict, metrics: dict, condition_rows: list[dict],
                            logits: np.ndarray, labels: np.ndarray, model_state: dict,
                            candidate: dict) -> None:
    if REQUIRED_METRICS - set(metrics):
        raise ValueError(f"Missing common metrics: {sorted(REQUIRED_METRICS - set(metrics))}")
    if metrics["selection_split"] != "validation" or metrics["final_test_evaluated"] is not False:
        raise ValueError("Artifact contract must be validation-only")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "run_config.json", config); atomic_json(output / "provenance.json", provenance())
    atomic_json(output / "metrics.json", metrics); atomic_json(output / "candidate.json", candidate)
    fields = sorted({key for row in condition_rows for key in row})
    with (output / "per_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(condition_rows)
    np.savez_compressed(output / "val_logits.npz", logits=logits, labels=labels)
    torch.save(model_state, output / "best_model.pt")
    atomic_json(output / "COMPLETED.json", {"status": "completed", "selection_split": "validation",
                                             "final_test_evaluated": False, "contract_version": 1})


def validate_completion(output: str | Path) -> dict:
    output = Path(output)
    missing = [name for name in REQUIRED_FILES if not (output / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete Phase-3 artifact contract: {missing}")
    completed = json.loads((output / "COMPLETED.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if completed.get("status") != "completed" or completed.get("selection_split") != "validation":
        raise ValueError("Invalid completion marker")
    if REQUIRED_METRICS - set(metrics) or metrics.get("selection_split") != "validation" or metrics.get("final_test_evaluated") is not False:
        raise ValueError("Invalid validation metrics contract")
    return {"completed": completed, "metrics": metrics}
