from __future__ import annotations

import numpy as np
import torch

from aigc_detector.data import ROBUSTNESS_CONDITIONS
from aigc_detector.metrics import classification_metrics


def summarize_validation(labels: torch.Tensor, probabilities: dict[str, torch.Tensor], threshold: float,
                         metadata: dict) -> tuple[dict, list[dict]]:
    if set(probabilities) != set(ROBUSTNESS_CONDITIONS):
        raise ValueError("Phase-3 scoring requires every exact Track-5 validation condition")
    per_condition = {name: classification_metrics(labels, probabilities[name], threshold)
                     for name in ROBUSTNESS_CONDITIONS}
    transformed = list(ROBUSTNESS_CONDITIONS[1:])
    baccs = [per_condition[name]["balanced_accuracy"] for name in transformed]
    aucs = [per_condition[name]["roc_auc"] for name in transformed]
    fprs = [per_condition[name]["false_positive_rate"] for name in transformed]
    worst = transformed[int(np.argmin(baccs))]
    clean = per_condition["clean"]
    summary = {
        **metadata,
        "clean_validation_balanced_accuracy": clean["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(np.min(baccs)),
        "worst_condition": worst,
        "resize_x0.25_balanced_accuracy": per_condition["resize_x0.25"]["balanced_accuracy"],
        "noise_sigma0.10_balanced_accuracy": per_condition["noise_s0.10"]["balanced_accuracy"],
        "blur_sigma2.0_balanced_accuracy": per_condition["blur_s2.0"]["balanced_accuracy"],
        "mean_transformed_roc_auc": float(np.mean(aucs)), "worst_transformed_roc_auc": float(np.min(aucs)),
        "clean_false_positive_rate": clean["false_positive_rate"],
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
        "selection_split": "validation", "final_test_evaluated": False,
    }
    return summary, [{"condition": name, **per_condition[name]} for name in ROBUSTNESS_CONDITIONS]
