from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEPLOYABLE_SOURCES = (
    ("E1b", "e1b_weight_sweep/weight_sweep_summary.json"),
    ("E1c", "e1c_ensemble/ensemble_summary.json"),
    ("E2b", "e2b_tta_search/tta_policy_summary.json"),
    ("E5_raw", "e5_quality_calibration/raw/calibration_summary.json"),
    ("E5_mild3", "e5_quality_calibration/mild3/calibration_summary.json"),
    ("E4b_fixed", "e4b_adaptive_fusion/fixed/summary.json"),
    ("E4b_global", "e4b_adaptive_fusion/global/summary.json"),
    ("E4b_adaptive", "e4b_adaptive_fusion/adaptive/summary.json"),
    ("E6", "e6_scale_consistency/validation_summary.json"),
    ("E7", "e7_radial_frequency/validation_summary.json"),
)

DIAGNOSTIC_SOURCES = (
    ("E4a", "e4a_ablation/ablation_summary.json"),
    ("E4c", "e4c_gate_intervention/intervention_summary.json"),
)

FIELDS = (
    "experiment", "variant", "experiment_type", "checkpoint_config_artifact",
    "clean_validation_balanced_accuracy", "mean_transformed_validation_balanced_accuracy",
    "worst_transformed_validation_balanced_accuracy", "worst_condition",
    "resize_x0.25_balanced_accuracy", "resize_x0.5_balanced_accuracy",
    "blur_sigma2.0_balanced_accuracy", "noise_sigma0.10_balanced_accuracy",
    "mean_transformed_roc_auc", "worst_transformed_roc_auc",
    "clean_false_positive_rate", "mean_transformed_false_positive_rate",
    "trainable_parameter_count", "inference_multiplier", "external_roc_auc",
    "external_recall", "clean_constraint_pass", "status", "validation_rank",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_completion_artifact(path: Path, stage: int) -> None:
    document = _read_json(path)
    if not isinstance(document, dict):
        raise ValueError(f"Stage {stage} completion artifact is not a JSON object: {path}")
    if stage != 0 and stage != 10:
        split = document.get("selection_split")
        if split is None and stage == 6:
            split = document.get("selection_split")
        if split != "validation":
            raise ValueError(f"Stage {stage} artifact is not validation-selected: {path}")
    if stage in {1, 2, 3, 4, 5, 7, 8, 9} and not document.get("results"):
        raise ValueError(f"Stage {stage} artifact has no results: {path}")
    if stage == 10:
        if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
            raise ValueError("Phase-2 recommendation is not a locked validation-only selection")


def _condition_value(row: dict, name: str) -> float | None:
    aliases = {
        "blur_sigma2.0": ("blur_sigma2.0", "blur_s2.0"),
        "noise_sigma0.10": ("noise_sigma0.10", "noise_s0.10"),
        "resize_x0.25": ("resize_x0.25",),
        "resize_x0.5": ("resize_x0.5",),
    }
    direct = row.get(f"{name}_balanced_accuracy")
    if direct is not None:
        return direct
    for container_name in ("per_condition_validation", "per_condition", "per_condition_balanced_accuracy"):
        container = row.get(container_name) or {}
        for alias in aliases[name]:
            value = container.get(alias)
            if isinstance(value, dict):
                value = value.get("balanced_accuracy")
            if value is not None:
                return value
    return None


def _variant(experiment: str, row: dict) -> str:
    keys = ("external_weight", "mode", "policy_name", "consistency_mode", "model_mode", "modality_subset")
    parts = [f"{key}={row[key]}" for key in keys if key in row]
    if "alpha" in row:
        parts.append(f"alpha={row['alpha']}")
    if "lambda_scale" in row:
        parts.append(f"lambda={row['lambda_scale']}")
    if "radial_bins" in row:
        parts.append(f"bins={row['radial_bins']}")
    return ",".join(parts) or experiment


def _artifact(experiment: str, row: dict, path: Path) -> str:
    if experiment.startswith("E1c"):
        return str(path.parent / "winning_ensemble.json")
    if experiment.startswith("E2b"):
        return str(path.parent / "winning_policy.json")
    if experiment.startswith("E5"):
        return str(path.parent / "winning_calibration.json")
    if experiment.startswith("E4b"):
        return str(path.parent / "model.pt")
    if experiment == "E6":
        return str(path.parent / "winning_config.json")
    if experiment == "E7":
        return str(path.parent / "winning_config.json")
    if row.get("checkpoint"):
        return str(row["checkpoint"])
    return str(path)


def normalize_candidate(experiment: str, row: dict, source: Path) -> dict:
    base_mode = "mild3" if experiment == "E5_mild3" else row.get("mode")
    multiplier = row.get("inference_multiplier") or (3 if base_mode == "mild3" else 1)
    if isinstance(multiplier, str) and multiplier.endswith("x"):
        multiplier = float(multiplier[:-1])
    clean_pass = row.get("clean_constraint_pass")
    if clean_pass is None and experiment.startswith("E4b"):
        clean_pass = (row.get("baseline_comparison") or {}).get("clean_balanced_accuracy_delta", -1) >= -0.01
    return {
        "experiment": experiment,
        "variant": _variant(experiment, row),
        "experiment_type": "deployment_candidate",
        "checkpoint_config_artifact": _artifact(experiment, row, source),
        "clean_validation_balanced_accuracy": row.get("clean_validation_balanced_accuracy"),
        "mean_transformed_validation_balanced_accuracy": row.get("mean_transformed_validation_balanced_accuracy"),
        "worst_transformed_validation_balanced_accuracy": row.get("worst_transformed_validation_balanced_accuracy"),
        "worst_condition": row.get("worst_condition"),
        "resize_x0.25_balanced_accuracy": _condition_value(row, "resize_x0.25"),
        "resize_x0.5_balanced_accuracy": _condition_value(row, "resize_x0.5"),
        "blur_sigma2.0_balanced_accuracy": _condition_value(row, "blur_sigma2.0"),
        "noise_sigma0.10_balanced_accuracy": _condition_value(row, "noise_sigma0.10"),
        "mean_transformed_roc_auc": row.get("mean_transformed_roc_auc"),
        "worst_transformed_roc_auc": row.get("worst_transformed_roc_auc"),
        "clean_false_positive_rate": row.get("clean_false_positive_rate"),
        "mean_transformed_false_positive_rate": row.get("mean_transformed_false_positive_rate"),
        "trainable_parameter_count": row.get("trainable_parameter_count"),
        "inference_multiplier": multiplier,
        "external_roc_auc": row.get("external_roc_auc"),
        "external_recall": row.get("external_recall"),
        "clean_constraint_pass": bool(clean_pass),
        "status": row.get("status", "succeeded"),
        "validation_rank": None,
    }


def rank_candidates(rows: list[dict]) -> list[dict]:
    eligible = [row for row in rows if row["status"] == "succeeded" and row["clean_constraint_pass"]]
    eligible.sort(key=lambda row: (
        row["worst_transformed_validation_balanced_accuracy"],
        row["mean_transformed_validation_balanced_accuracy"],
        -float(row.get("inference_multiplier") or 1),
        -float(row.get("trainable_parameter_count") or 0),
    ), reverse=True)
    ranks = {(row["experiment"], row["variant"]): index for index, row in enumerate(eligible, 1)}
    return [{**row, "validation_rank": ranks.get((row["experiment"], row["variant"]))} for row in rows]


def aggregate(root: Path) -> dict:
    candidates: list[dict] = []
    diagnostics = []
    for experiment, relative in DEPLOYABLE_SOURCES:
        path = root / relative
        document = _read_json(path)
        if document.get("selection_split") != "validation":
            raise ValueError(f"{experiment} summary is not validation-selected")
        rows = document.get("results") if "results" in document else [document]
        candidates.extend(normalize_candidate(experiment, row, path) for row in rows)
    for experiment, relative in DIAGNOSTIC_SOURCES:
        path = root / relative
        document = _read_json(path)
        if document.get("selection_split") != "validation":
            raise ValueError(f"{experiment} diagnostic is not validation-only")
        diagnostics.append({"experiment": experiment, "experiment_type": "diagnostic", "source": str(path),
                            "results": document.get("results", [])})
    stability_path = root / "e7_radial_frequency/stability_by_scale.csv"
    if stability_path.is_file():
        diagnostics.append({"experiment": "E7_stability", "experiment_type": "diagnostic",
                            "source": str(stability_path), "excluded_from_deployment_ranking": True})
    ranked = rank_candidates(candidates)
    winner = min((row for row in ranked if row["validation_rank"] is not None), key=lambda row: row["validation_rank"])
    output = root / "phase2"
    output.mkdir(parents=True, exist_ok=True)
    document = {"selection_split": "validation", "final_test_evaluated": False,
                "external_metrics_used_for_ranking": False,
                "missing_metrics_policy": "Unavailable metrics are stored as JSON null and blank CSV cells; they are never inferred.",
                "results": ranked}
    (output / "phase2_summary.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with (output / "phase2_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(ranked)
    (output / "diagnostic_summary.json").write_text(json.dumps({
        "selection_split": "validation", "excluded_from_deployment_ranking": True, "diagnostics": diagnostics,
    }, indent=2) + "\n", encoding="utf-8")
    recommendation = {
        "experiment": winner["experiment"], "variant": winner["variant"],
        "checkpoint_config_paths": [winner["checkpoint_config_artifact"]],
        "validation_metrics": {key: winner[key] for key in (
            "clean_validation_balanced_accuracy", "mean_transformed_validation_balanced_accuracy",
            "worst_transformed_validation_balanced_accuracy", "worst_condition")},
        "why_it_won": "Highest worst transformed validation balanced accuracy among clean-eligible deployable candidates; ties use mean transformed validation balanced accuracy, then lower inference cost and complexity.",
        "inference_multiplier": winner["inference_multiplier"],
        "trainable_parameter_count": winner["trainable_parameter_count"],
        "clean_constraint_pass": winner["clean_constraint_pass"],
        "selection_split": "validation", "final_test_evaluated": False,
    }
    (output / "recommended_candidate.json").write_text(json.dumps(recommendation, indent=2) + "\n", encoding="utf-8")
    with (output / "pareto_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("candidate", "worst_val_bacc", "mean_val_bacc", "external_auc", "external_recall", "inference_multiplier", "params")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in ranked:
            writer.writerow({"candidate": f"{row['experiment']}:{row['variant']}",
                             "worst_val_bacc": row["worst_transformed_validation_balanced_accuracy"],
                             "mean_val_bacc": row["mean_transformed_validation_balanced_accuracy"],
                             "external_auc": row["external_roc_auc"], "external_recall": row["external_recall"],
                             "inference_multiplier": row["inference_multiplier"], "params": row["trainable_parameter_count"]})
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate validation-only Track-5 Phase-2 research outputs")
    sub = parser.add_subparsers(dest="command", required=True)
    aggregate_parser = sub.add_parser("aggregate"); aggregate_parser.add_argument("--track5-root", type=Path, required=True)
    validate = sub.add_parser("validate-artifact"); validate.add_argument("--path", type=Path, required=True); validate.add_argument("--stage", type=int, required=True)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.track5_root)
    else:
        validate_completion_artifact(args.path, args.stage)


if __name__ == "__main__":
    main()
