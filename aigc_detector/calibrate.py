"""Fit temperature and operating thresholds on the reserved calibration split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .data import ExactSeverityTransform, SEVERITY_SPECS, load_split_manifest, severity_key
from .features import extract_features_with_factory
from .metrics import (
    classification_metrics,
    expected_calibration_error,
    fit_temperature,
    operational_thresholds,
    select_threshold,
)
from .model import FrozenEncoders, load_checkpoint, save_checkpoint
from .train import choose_device


def _metrics(labels: torch.Tensor, logits: torch.Tensor, temperature: float) -> dict[str, float]:
    probabilities = torch.sigmoid(logits / temperature)
    result = classification_metrics(labels, probabilities)
    result["ece"] = expected_calibration_error(labels, probabilities)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit clean-only and mixed-condition calibration on the reserved calibration split")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True, help="Scale cache containing clean calibration features")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--selection", choices=("clean", "mixed"), default="mixed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--precision-target", type=float, default=0.95)
    args = parser.parse_args()

    device = choose_device(args.device)
    head, config, old_temperature, metadata = load_checkpoint(args.checkpoint, device)
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    # Calibration never reuses model-selection or test rows.
    clean_features = cache.get("calibration_features")
    clean_labels = cache.get("calibration_labels")
    if clean_features is None or clean_labels is None:
        raise ValueError("Feature cache is missing the separate calibration split")
    expected_width = config.clip_dim + config.forensic_dim
    if clean_features.shape[1] != expected_width:
        raise ValueError(f"Calibration cache width {clean_features.shape[1]} does not match checkpoint width {expected_width}")
    calibration_rows = load_split_manifest(args.data_dir, args.split_manifest)["calibration"]
    if len(calibration_rows) != len(clean_labels):
        raise ValueError("Calibration cache row count does not match the split manifest")
    if not torch.equal(clean_labels, torch.tensor([label for _, label in calibration_rows], dtype=torch.float32)):
        raise ValueError("Calibration cache labels do not align with manifest order")

    # Assign exact severities evenly so one corruption family cannot dominate.
    transformed_specs = [spec for spec in SEVERITY_SPECS if spec[0] != "clean"]
    assignments = [transformed_specs[index % len(transformed_specs)] for index in range(len(calibration_rows))]
    transformed_features, transformed_labels, _ = extract_features_with_factory(
        calibration_rows,
        FrozenEncoders(config, device),
        args.batch_size,
        lambda path, index: ExactSeverityTransform(*assignments[index], seed=args.seed, key=str(path)),
        "calibration transformed mixture",
    )
    if not torch.equal(clean_labels, transformed_labels):
        raise ValueError("Transformed calibration labels changed order")
    with torch.no_grad():
        clean_logits = head(clean_features.to(device)).cpu()
        transformed_logits = head(transformed_features.to(device)).cpu()
    mixed_logits = torch.cat((clean_logits, transformed_logits))
    mixed_labels = torch.cat((clean_labels, transformed_labels))
    clean_temperature = fit_temperature(clean_logits, clean_labels)
    mixed_temperature = fit_temperature(mixed_logits, mixed_labels)
    chosen_temperature = clean_temperature if args.selection == "clean" else mixed_temperature
    chosen_probabilities = torch.sigmoid(mixed_logits / chosen_temperature)
    selected_threshold = select_threshold(mixed_labels, chosen_probabilities, "balanced")
    thresholds = operational_thresholds(
        mixed_labels, chosen_probabilities, args.recall_target, args.precision_target
    )
    report = {
        "_metadata": {
            "input_checkpoint": str(args.checkpoint),
            "feature_cache": str(args.feature_cache),
            "split_manifest": str(args.split_manifest),
            "calibration_originals": len(clean_labels),
            "mixed_calibration_rows": len(mixed_labels),
            "seed": args.seed,
            "old_temperature": old_temperature,
            "selected_policy": args.selection,
            "assignment_counts": dict(Counter(severity_key(*spec) for spec in assignments)),
        },
        "temperatures": {"clean": clean_temperature, "mixed": mixed_temperature, "selected": chosen_temperature},
        "selected_threshold": selected_threshold,
        "comparison": {
            policy: {
                "clean": _metrics(clean_labels, clean_logits, temperature),
                "transformed": _metrics(transformed_labels, transformed_logits, temperature),
                "mixed": _metrics(mixed_labels, mixed_logits, temperature),
                "by_transformed_condition": {
                    key: _metrics(
                        transformed_labels[torch.tensor([severity_key(*spec) == key for spec in assignments])],
                        transformed_logits[torch.tensor([severity_key(*spec) == key for spec in assignments])],
                        temperature,
                    )
                    for key in sorted({severity_key(*spec) for spec in assignments})
                },
            }
            for policy, temperature in (("clean", clean_temperature), ("mixed", mixed_temperature))
        },
        "operational_thresholds": thresholds,
    }
    output_metadata = {
        **metadata,
        "calibration": report,
        "operational_thresholds": thresholds,
        "threshold": selected_threshold,
        "threshold_objective": "balanced_mixed_calibration",
    }
    head.to("cpu")
    save_checkpoint(args.output_checkpoint, head, config, chosen_temperature, output_metadata)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output_checkpoint), **report}, indent=2))


if __name__ == "__main__":
    main()
