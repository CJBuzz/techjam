from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import (
    ROBUSTNESS_CONDITIONS,
    ROBUST_SELECTION_CONDITIONS,
    image_source,
    load_labeled_paths,
    stratified_train_val_test_split,
)
from .features import extract_condition_features
from .metrics import classification_metrics
from .model import AdaptiveTriExpertHead, ExpertMixtureHead, FrozenEncoders, load_checkpoint
from .train import choose_device


def paired_generator_metrics(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    sources: list[str],
    threshold: float,
) -> dict[str, object]:
    """Pair every synthetic generator with the shared real set.

    This avoids reporting a misleading 1:2-class-ratio accuracy for B-Free's
    one real source and two synthetic generators. The checkpoint threshold is
    kept frozen; no external labels are used for calibration or tuning.
    """
    labels = labels.detach().cpu()
    probabilities = probabilities.detach().cpu()
    real_mask = labels == 0
    generators = sorted({source for source, label in zip(sources, labels.tolist()) if int(label) == 1})
    if not bool(real_mask.any()) or not generators:
        raise ValueError("Paired-generator evaluation requires real images and at least one AI source")
    source_reports: dict[str, dict[str, object]] = {}
    for generator in generators:
        fake_mask = torch.tensor(
            [source == generator and int(label) == 1 for source, label in zip(sources, labels.tolist())],
            dtype=torch.bool,
        )
        pair_mask = real_mask | fake_mask
        report = classification_metrics(labels[pair_mask], probabilities[pair_mask], threshold)
        report["real_images"] = int(real_mask.sum())
        report["synthetic_images"] = int(fake_mask.sum())
        source_reports[generator] = report
    balanced = {name: float(report["balanced_accuracy"]) for name, report in source_reports.items()}
    aucs = {name: float(report["roc_auc"]) for name, report in source_reports.items()}
    real_predicted_ai = probabilities[real_mask] >= threshold
    return {
        "real_images": int(real_mask.sum()),
        "real_false_positive_rate": float(real_predicted_ai.float().mean()),
        "real_mean_probability": float(probabilities[real_mask].mean()),
        "generators": source_reports,
        "macro_balanced_accuracy": float(np.mean(list(balanced.values()))),
        "macro_roc_auc": float(np.mean(list(aucs.values()))),
        "worst_generator_balanced_accuracy": float(min(balanced.values())),
        "worst_generator": min(balanced, key=balanced.get),
    }


def source_diagnostics(
    labels: torch.Tensor, probabilities: torch.Tensor, sources: list[str], threshold: float
) -> dict[str, dict[str, float | int]]:
    """Report class-aware diagnostics for sources that may contain one class only."""
    reports = {}
    for source in sorted(set(sources)):
        mask = torch.tensor([item == source for item in sources], dtype=torch.bool)
        source_labels = labels[mask]
        source_probabilities = probabilities[mask]
        reports[source] = {
            "images": int(mask.sum()),
            "true_ai_fraction": float(source_labels.mean()),
            "predicted_ai_fraction": float((source_probabilities >= threshold).float().mean()),
            "mean_probability": float(source_probabilities.mean()),
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clean and transformed-image robustness")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        dest="checkpoints",
        help="Checkpoint to evaluate; repeat for compatible checkpoints to reuse extracted features",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness.json"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "all"),
        default="test",
        help="Use validation for candidates, test for the final model, or all for a separate external dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile", choices=("full", "worst"), default="full",
        help="Full scores every official severity; worst is a faster hardest-severity check",
    )
    parser.add_argument("--error-analysis-output", type=Path, default=None)
    parser.add_argument("--top-errors", type=int, default=12)
    parser.add_argument(
        "--protocol",
        choices=("standard", "paired-generators"),
        default="standard",
        help=(
            "Paired-generators evaluates each AI source against the shared real set and "
            "macro-averages generators; use it with --split all for external benchmarks"
        ),
    )
    args = parser.parse_args()
    if args.protocol == "paired-generators" and args.split != "all":
        raise ValueError("--protocol paired-generators requires --split all")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint_paths = args.checkpoints or [Path("artifacts/hybrid_detector.pt")]
    models = []
    for checkpoint_path in checkpoint_paths:
        head, config, temperature, checkpoint_metadata = load_checkpoint(checkpoint_path, device)
        models.append((str(checkpoint_path), head, config, temperature, checkpoint_metadata))
    reference_config = models[0][2]
    encoder_fields = ("clip_model", "clip_dim", "forensic_dim", "forensic_mode")
    reference_signature = tuple(getattr(reference_config, field) for field in encoder_fields)
    if any(tuple(getattr(config, field) for field in encoder_fields) != reference_signature for _, _, config, _, _ in models[1:]):
        raise ValueError("All checkpoints in one evaluation must use the same encoder configuration")
    encoder_config = reference_config
    if any(config.quality_dim for _, _, config, _, _ in models):
        from dataclasses import replace
        encoder_config = replace(reference_config, quality_dim=6)
    encoders = FrozenEncoders(encoder_config, device)
    all_rows = load_labeled_paths(args.data_dir)
    if args.split == "all":
        rows = all_rows
    else:
        _, validation_rows, test_rows = stratified_train_val_test_split(
            all_rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
        )
        rows = validation_rows if args.split == "validation" else test_rows
    print(f"Evaluating untouched {args.split} split: {len(rows)} of {len(all_rows)} source images")
    metadata = {
        "total_source_images": len(all_rows),
        "split": args.split,
        "split_images": len(rows),
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "transform_strengths": "every official challenge severity is evaluated deterministically",
        "protocol": args.protocol,
        "external_threshold_policy": (
            "checkpoint calibration and threshold are frozen; external labels are used only for scoring"
            if args.protocol == "paired-generators" else None
        ),
    }
    model_results = {path: {} for path, _, _, _, _ in models}
    error_records = {path: [] for path, _, _, _, _ in models}
    conditions = ROBUSTNESS_CONDITIONS if args.profile == "full" else ("clean", *ROBUST_SELECTION_CONDITIONS)
    metadata["conditions"] = list(conditions)
    for name in conditions:
        features, labels, paths, _ = extract_condition_features(
            rows, encoders, args.batch_size, (name,), args.seed
        )
        sources = [image_source(path, args.data_dir) for path in paths]
        for checkpoint_path, head, model_config, temperature, checkpoint_metadata in models:
            model_features = features if model_config.quality_dim else features[:, : model_config.clip_dim + model_config.forensic_dim]
            threshold = float(checkpoint_metadata.get("threshold", 0.5))
            with torch.no_grad():
                probabilities = torch.sigmoid(head(model_features.to(device)) / temperature).cpu()
            condition_result: dict[str, object] = {
                "overall": classification_metrics(labels, probabilities, threshold)
            }
            if args.protocol == "paired-generators":
                condition_result["source_diagnostics"] = source_diagnostics(
                    labels, probabilities, sources, threshold
                )
                condition_result["paired_generators"] = paired_generator_metrics(
                    labels, probabilities, sources, threshold
                )
            else:
                by_source = {}
                for source in sorted(set(sources)):
                    mask = torch.tensor([item == source for item in sources], dtype=torch.bool)
                    by_source[source] = classification_metrics(labels[mask], probabilities[mask], threshold)
                condition_result["by_source"] = by_source
            model_results[checkpoint_path][name] = condition_result
            predictions = probabilities >= threshold
            for index, (truth, prediction) in enumerate(zip(labels.bool(), predictions)):
                if bool(truth) != bool(prediction):
                    error_records[checkpoint_path].append({
                        "condition": name,
                        "image_path": paths[index],
                        "source": sources[index],
                        "label": int(labels[index]),
                        "pred": float(probabilities[index]),
                        "error_type": "false_negative" if bool(truth) else "false_positive",
                    })
            if isinstance(head, (ExpertMixtureHead, AdaptiveTriExpertHead)):
                with torch.no_grad(): gate = head.gate_weights(model_features.to(device)).cpu()
                gate_means = gate.mean(0)
                model_results[checkpoint_path][name]["expert_gate"] = {
                    "overall_mean": gate_means.tolist() if gate_means.ndim else float(gate_means),
                    "by_source_mean": {
                        source: gate[torch.tensor([item == source for item in sources])].mean(0).tolist()
                        if gate.ndim == 2 else float(gate[torch.tensor([item == source for item in sources])].mean())
                        for source in sorted(set(sources))
                    },
                }
    for checkpoint_path in model_results:
        if args.protocol == "paired-generators":
            condition_scores = {
                condition: float(values["paired_generators"]["macro_balanced_accuracy"])
                for condition, values in model_results[checkpoint_path].items()
            }
            clean_score = condition_scores["clean"]
            transformed = {name: value for name, value in condition_scores.items() if name != "clean"}
            model_results[checkpoint_path]["_generalization_summary"] = {
                "metric": "macro_generator_balanced_accuracy",
                "clean_score": clean_score,
                "mean_transformed_score": float(np.mean(list(transformed.values()))),
                "worst_transformed_score": float(min(transformed.values())),
                "worst_condition": min(transformed, key=transformed.get),
                "mean_drop_from_clean": float(clean_score - np.mean(list(transformed.values()))),
            }
        else:
            condition_accuracies = {
                condition: values["overall"]["accuracy"]
                for condition, values in model_results[checkpoint_path].items()
            }
            clean_accuracy = condition_accuracies["clean"]
            transformed = {name: value for name, value in condition_accuracies.items() if name != "clean"}
            model_results[checkpoint_path]["_robust_summary"] = {
                "clean_accuracy": clean_accuracy,
                "mean_transformed_accuracy": float(np.mean(list(transformed.values()))),
                "worst_transformed_accuracy": float(min(transformed.values())),
                "worst_condition": min(transformed, key=transformed.get),
                "mean_drop_from_clean": float(clean_accuracy - np.mean(list(transformed.values()))),
            }
        ranked = sorted(
            error_records[checkpoint_path],
            key=lambda item: item["pred"] if item["error_type"] == "false_positive" else 1 - item["pred"],
            reverse=True,
        )
        error_records[checkpoint_path] = ranked[: args.top_errors]
    if len(models) == 1:
        results = {
            "_metadata": {**metadata, "checkpoint": models[0][0]},
            **model_results[models[0][0]],
        }
    else:
        results = {"_metadata": metadata, "checkpoints": model_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.error_analysis_output:
        args.error_analysis_output.parent.mkdir(parents=True, exist_ok=True)
        errors = error_records[models[0][0]] if len(models) == 1 else error_records
        args.error_analysis_output.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
