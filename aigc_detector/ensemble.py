from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import load_labeled_paths
from .features import extract_features
from .metrics import classification_metrics
from .model import FrozenEncoders, load_checkpoint
from .train import choose_device


ENCODER_FIELDS = ("clip_model", "clip_dim", "forensic_dim", "forensic_mode")


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Dataset must be written as NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Dataset must contain a non-empty name and path")
    return name.strip(), Path(raw_path.strip())


def compatible_encoder_config(configs):
    reference = configs[0]
    signature = tuple(getattr(reference, field) for field in ENCODER_FIELDS)
    if any(tuple(getattr(config, field) for field in ENCODER_FIELDS) != signature for config in configs[1:]):
        raise ValueError("All ensemble checkpoints must use the same encoder configuration")
    return replace(reference, quality_dim=max(config.quality_dim for config in configs))


def score_head(features: torch.Tensor, head, config, temperature: float, device: torch.device) -> torch.Tensor:
    model_features = features if config.quality_dim else features[:, : config.clip_dim + config.forensic_dim]
    with torch.no_grad():
        return torch.sigmoid(head(model_features.to(device)) / temperature).cpu()


def _rates(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real = np.sort(probabilities[labels == 0])
    fake = np.sort(probabilities[labels == 1])
    if not len(real) or not len(fake):
        raise ValueError("Every selection dataset must contain both real and synthetic images")
    false_positive_rate = (len(real) - np.searchsorted(real, thresholds, side="left")) / len(real)
    true_positive_rate = (len(fake) - np.searchsorted(fake, thresholds, side="left")) / len(fake)
    return false_positive_rate, true_positive_rate


def select_ensemble_policy(
    datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    weights: list[float],
    max_real_fpr: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Select a two-head probability blend and one global development threshold.

    Each dataset contributes equally, regardless of its image count. The real
    false-positive constraint must hold independently on every dataset.
    """
    if not datasets:
        raise ValueError("At least one development dataset is required")
    if not 0 <= max_real_fpr <= 1:
        raise ValueError("max_real_fpr must be between 0 and 1")
    rows: list[dict[str, float]] = []
    for weight_40k in weights:
        if not 0 <= weight_40k <= 1:
            raise ValueError("Ensemble weights must be between 0 and 1")
        blended = {
            name: (labels, weight_40k * first + (1 - weight_40k) * second)
            for name, (labels, first, second) in datasets.items()
        }
        candidates = np.unique(np.concatenate([
            np.array([0.0, 0.5, 1.0]),
            *(probabilities for _, probabilities in blended.values()),
        ]))
        fprs, baccs = [], []
        aucs = []
        for labels, probabilities in blended.values():
            fpr, tpr = _rates(labels, probabilities, candidates)
            fprs.append(fpr)
            baccs.append((1 - fpr + tpr) / 2)
            aucs.append(float(roc_auc_score(labels, probabilities)))
        fpr_matrix = np.stack(fprs)
        bacc_matrix = np.stack(baccs)
        feasible = np.max(fpr_matrix, axis=0) <= max_real_fpr + 1e-12
        if not bool(np.any(feasible)):
            raise ValueError("No threshold satisfies the requested real false-positive limit")
        macro_bacc = np.mean(bacc_matrix, axis=0)
        worst_bacc = np.min(bacc_matrix, axis=0)
        feasible_indices = np.flatnonzero(feasible)
        best_index = max(
            feasible_indices,
            key=lambda index: (
                macro_bacc[index],
                worst_bacc[index],
                -abs(float(candidates[index]) - 0.5),
            ),
        )
        rows.append({
            "weight_40k": float(weight_40k),
            "weight_100k": float(1 - weight_40k),
            "threshold": float(candidates[best_index]),
            "macro_balanced_accuracy": float(macro_bacc[best_index]),
            "worst_dataset_balanced_accuracy": float(worst_bacc[best_index]),
            "worst_dataset_real_fpr": float(np.max(fpr_matrix[:, best_index])),
            "macro_roc_auc": float(np.mean(aucs)),
        })
    selected = max(
        rows,
        key=lambda row: (
            row["macro_balanced_accuracy"],
            row["worst_dataset_balanced_accuracy"],
            row["macro_roc_auc"],
            -abs(row["weight_40k"] - 0.5),
        ),
    )
    return selected, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a 40K/100K ensemble on external development sets")
    parser.add_argument("--dataset", action="append", type=parse_dataset, default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--predictions-input", action="append", type=Path, default=[],
        help="Reuse per-image scores written by an earlier --predictions-output run",
    )
    parser.add_argument("--checkpoint-40k", type=Path, required=True)
    parser.add_argument("--checkpoint-100k", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[index / 20 for index in range(21)])
    parser.add_argument("--max-real-fpr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, default=None)
    args = parser.parse_args()

    if not args.dataset and not args.predictions_input:
        raise ValueError("Provide at least one --dataset or --predictions-input")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoints = [args.checkpoint_40k, args.checkpoint_100k]
    loaded = [load_checkpoint(path, device) for path in checkpoints]
    selection_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    prediction_records: list[dict[str, object]] = []
    for predictions_path in args.predictions_input:
        records = json.loads(predictions_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict[str, object]]] = {}
        for record in records:
            grouped.setdefault(str(record["dataset"]), []).append(record)
        for dataset_name, dataset_records in grouped.items():
            if dataset_name in selection_data:
                raise ValueError(f"Duplicate dataset name {dataset_name!r} in prediction inputs")
            selection_data[dataset_name] = (
                np.asarray([record["label"] for record in dataset_records], dtype=int),
                np.asarray([record["probability_40k"] for record in dataset_records], dtype=float),
                np.asarray([record["probability_100k"] for record in dataset_records], dtype=float),
            )
            prediction_records.extend(dataset_records)

    configs = [item[1] for item in loaded]
    encoders = FrozenEncoders(compatible_encoder_config(configs), device) if args.dataset else None
    for dataset_name, data_dir in args.dataset:
        if dataset_name in selection_data:
            raise ValueError(f"Duplicate dataset name {dataset_name!r}")
        rows = load_labeled_paths(data_dir)
        if not rows:
            raise ValueError(f"No labeled images found under {data_dir}")
        features, labels, paths = extract_features(rows, encoders, args.batch_size)
        probabilities = [
            score_head(features, head, config, temperature, device)
            for head, config, temperature, _ in loaded
        ]
        label_values = labels.numpy().astype(int)
        selection_data[dataset_name] = (label_values, probabilities[0].numpy(), probabilities[1].numpy())
        prediction_records.extend({
            "dataset": dataset_name,
            "image_path": path,
            "label": int(label),
            "probability_40k": float(probabilities[0][index]),
            "probability_100k": float(probabilities[1][index]),
        } for index, (path, label) in enumerate(zip(paths, label_values)))

    frozen_reports: dict[str, dict[str, object]] = {}
    for dataset_name, (labels, first, second) in selection_data.items():
        label_tensor = torch.from_numpy(labels)
        frozen_reports[dataset_name] = {
            "40k": classification_metrics(
                label_tensor, torch.from_numpy(first), float(loaded[0][3].get("threshold", 0.5))
            ),
            "100k": classification_metrics(
                label_tensor, torch.from_numpy(second), float(loaded[1][3].get("threshold", 0.5))
            ),
        }

    selected, sweep = select_ensemble_policy(selection_data, args.weights, args.max_real_fpr)
    selected_reports = {}
    for dataset_name, (labels, first, second) in selection_data.items():
        blended = selected["weight_40k"] * first + selected["weight_100k"] * second
        selected_reports[dataset_name] = classification_metrics(
            torch.from_numpy(labels), torch.from_numpy(blended), selected["threshold"]
        )
    result = {
        "policy": {
            "checkpoint_40k": str(args.checkpoint_40k),
            "checkpoint_100k": str(args.checkpoint_100k),
            **selected,
            "max_real_fpr": args.max_real_fpr,
        },
        "selection": {
            "datasets": {
                **{name: "precomputed predictions" for name in selection_data},
                **{name: str(path) for name, path in args.dataset},
            },
            "seed": args.seed,
            "objective": "macro balanced accuracy with an independent per-dataset real-FPR cap",
            "status": "development-only; all listed datasets are model-selection data",
        },
        "frozen_checkpoint_reports": frozen_reports,
        "selected_ensemble_reports": selected_reports,
        "weight_sweep": sweep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        for record in prediction_records:
            record["ensemble_probability"] = (
                selected["weight_40k"] * record["probability_40k"]
                + selected["weight_100k"] * record["probability_100k"]
            )
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(json.dumps(prediction_records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
