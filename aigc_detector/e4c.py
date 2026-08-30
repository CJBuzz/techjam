from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .data import ROBUSTNESS_CONDITIONS
from .e4a import assemble_validation_matrix, prepare_missing_validation_features
from .e4b import AdaptiveFusionHead, load_adaptive_checkpoint
from .metrics import classification_metrics


INTERVENTION_WEIGHTS = OrderedDict((
    ("equal", (1 / 3, 1 / 3, 1 / 3)),
    ("semantic_heavy", (0.60, 0.20, 0.20)),
    ("forensic_heavy", (0.20, 0.40, 0.40)),
    ("laplacian_heavy", (0.20, 0.60, 0.20)),
    ("fft_heavy", (0.20, 0.20, 0.60)),
))
INTERVENTION_MODES = ("learned", "equal", "global_mean", *tuple(INTERVENTION_WEIGHTS)[1:])


def validate_interventions() -> None:
    for name, weights in INTERVENTION_WEIGHTS.items():
        if len(weights) != 3 or not np.isclose(sum(weights), 1.0) or min(weights) < 0:
            raise ValueError(f"Invalid fixed gate intervention: {name}")


def intervention_logits(
    model: AdaptiveFusionHead,
    features: torch.Tensor,
    intervention: str,
    clean_mean_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace only gate weights; no condition or label is accepted as an input."""
    validate_interventions()
    modality_logits = model.modality_logits(features)
    if intervention == "learned":
        weights = model.gate_weights(features, modality_logits)
    elif intervention == "global_mean":
        if clean_mean_weights is None or clean_mean_weights.shape != (3,):
            raise ValueError("global_mean requires one three-element clean validation mean")
        weights = clean_mean_weights.to(features).expand(len(features), -1)
    elif intervention in INTERVENTION_WEIGHTS:
        weights = torch.tensor(INTERVENTION_WEIGHTS[intervention], dtype=features.dtype,
                               device=features.device).expand(len(features), -1)
    else:
        raise ValueError(f"Unknown gate intervention: {intervention}")
    return (weights * modality_logits).sum(1), weights


def learned_gate_shifts(
    clean_weights: torch.Tensor, robust_weights: torch.Tensor, robust_conditions: list[str]
) -> list[dict]:
    clean_mean = clean_weights.mean(0)
    rows = [{
        "condition": "clean", "semantic_weight": float(clean_mean[0]),
        "laplacian_weight": float(clean_mean[1]), "fft_weight": float(clean_mean[2]),
        "l1_distance_from_clean": 0.0, "semantic_weight_change": 0.0,
        "laplacian_weight_change": 0.0, "fft_weight_change": 0.0,
    }]
    for condition in ROBUSTNESS_CONDITIONS[1:]:
        mask = torch.tensor([value == condition for value in robust_conditions])
        mean = robust_weights[mask].mean(0)
        delta = mean - clean_mean
        rows.append({
            "condition": condition, "semantic_weight": float(mean[0]),
            "laplacian_weight": float(mean[1]), "fft_weight": float(mean[2]),
            "l1_distance_from_clean": float(delta.abs().sum()),
            "semantic_weight_change": float(delta[0]),
            "laplacian_weight_change": float(delta[1]), "fft_weight_change": float(delta[2]),
        })
    return rows


def score_intervention(
    model: AdaptiveFusionHead, intervention: str,
    clean_x: torch.Tensor, clean_y: torch.Tensor,
    robust_x: torch.Tensor, robust_y: torch.Tensor, robust_conditions: list[str],
    temperature: float, threshold: float, clean_mean_weights: torch.Tensor,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        clean_logits, clean_weights = intervention_logits(model, clean_x, intervention, clean_mean_weights)
        robust_logits, robust_weights = intervention_logits(model, robust_x, intervention, clean_mean_weights)
    clean_metrics = classification_metrics(clean_y, torch.sigmoid(clean_logits / temperature), threshold)
    robust_probabilities = torch.sigmoid(robust_logits / temperature)
    per_condition = {}
    for condition in ROBUSTNESS_CONDITIONS[1:]:
        mask = torch.tensor([value == condition for value in robust_conditions])
        per_condition[condition] = classification_metrics(
            robust_y[mask], robust_probabilities[mask], threshold
        )
    transformed = list(ROBUSTNESS_CONDITIONS[1:])
    baccs = [per_condition[name]["balanced_accuracy"] for name in transformed]
    fprs = [per_condition[name]["false_positive_rate"] for name in transformed]
    worst_index = int(np.argmin(baccs))
    explicit_weights = (
        "learned_per_sample" if intervention == "learned" else
        clean_mean_weights.tolist() if intervention == "global_mean" else
        list(INTERVENTION_WEIGHTS[intervention])
    )
    return ({
        "intervention": intervention, "weights_or_mode": explicit_weights,
        "deployment_capable": True,
        "clean_validation_balanced_accuracy": clean_metrics["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
        "worst_condition": transformed[worst_index],
        "per_condition_balanced_accuracy": {
            "clean": clean_metrics["balanced_accuracy"],
            **{name: per_condition[name]["balanced_accuracy"] for name in transformed},
        },
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
        "temperature_and_threshold_refit": False,
    }, clean_weights, robust_weights)


def add_deltas(rows: list[dict]) -> list[dict]:
    learned = next(row for row in rows if row["intervention"] == "learned")
    equal = next(row for row in rows if row["intervention"] == "equal")
    global_mean = next(row for row in rows if row["intervention"] == "global_mean")
    metrics = (
        "clean_validation_balanced_accuracy", "mean_transformed_validation_balanced_accuracy",
        "worst_transformed_validation_balanced_accuracy",
    )
    return [{
        **row,
        "delta_vs_learned": {key: row[key] - learned[key] for key in metrics},
        "delta_vs_equal": {key: row[key] - equal[key] for key in metrics},
        "delta_vs_global_mean": {key: row[key] - global_mean[key] for key in metrics},
    } for row in rows]


def write_outputs(rows: list[dict], shifts: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "experiment": "E4c controlled gate intervention diagnostic",
        "selection_split": "validation", "test_rows_used": False,
        "condition_labels_used_only_for_post_hoc_reporting": True,
        "results": rows, "learned_gate_shifts": shifts,
    }
    (output_dir / "intervention_summary.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    scalar_keys = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with (output_dir / "intervention_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys + [
            "weights_or_mode", "per_condition_balanced_accuracy", "delta_vs_learned",
            "delta_vs_equal", "delta_vs_global_mean",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys} | {
                key: json.dumps(row[key], sort_keys=True) for key in (
                    "weights_or_mode", "per_condition_balanced_accuracy", "delta_vs_learned",
                    "delta_vs_equal", "delta_vs_global_mean",
                )
            })
    with (output_dir / "condition_gate_shift.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shifts[0]))
        writer.writeheader()
        writer.writerows(shifts)


def main() -> None:
    parser = argparse.ArgumentParser(description="E4c controlled interventions on an E4b gate")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    model, temperature, threshold, metadata = load_adaptive_checkpoint(args.model, torch.device("cpu"))
    if model.mode != "adaptive":
        raise ValueError("E4c learned intervention requires an adaptive E4b checkpoint")
    if metadata.get("selection_split") != "validation" or metadata.get("test_rows_used_for_selection") is not False:
        raise ValueError("E4b checkpoint was not selected exclusively on validation")
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True, mmap=True)
    extra = prepare_missing_validation_features(
        base, args.base_cache, args.validation_cache, args.data_dir, args.device, args.feature_batch_size
    )
    robust_x, robust_y, robust_conditions = assemble_validation_matrix(base, extra)
    clean_x, clean_y = base["val_features"], base["val_labels"]
    with torch.no_grad():
        learned_clean_weights = model.gate_weights(clean_x)
        learned_robust_weights = model.gate_weights(robust_x)
    clean_mean = learned_clean_weights.mean(0)
    rows = [score_intervention(
        model, intervention, clean_x, clean_y, robust_x, robust_y, robust_conditions,
        temperature, threshold, clean_mean,
    )[0] for intervention in INTERVENTION_MODES]
    rows = add_deltas(rows)
    shifts = learned_gate_shifts(learned_clean_weights, learned_robust_weights, robust_conditions)
    write_outputs(rows, shifts, args.output_dir)


if __name__ == "__main__":
    main()
