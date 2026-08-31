from __future__ import annotations

import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..data import ROBUSTNESS_CONDITIONS
from ..metrics import classification_metrics, fit_temperature, select_threshold


FEATURE_BLOCKS = OrderedDict((
    ("clip", slice(0, 512)),
    ("laplacian", slice(512, 1792)),
    ("fft", slice(1792, 3072)),
))
MODALITY_SUBSETS = OrderedDict((
    ("clip", ("clip",)),
    ("laplacian", ("laplacian",)),
    ("fft", ("fft",)),
    ("laplacian+fft", ("laplacian", "fft")),
    ("clip+laplacian", ("clip", "laplacian")),
    ("clip+fft", ("clip", "fft")),
    ("clip+laplacian+fft", ("clip", "laplacian", "fft")),
))
OFFICIAL_TRANSFORMED_CONDITIONS = tuple(c for c in ROBUSTNESS_CONDITIONS if c != "clean")


def validate_feature_layout(width: int = 3072) -> None:
    occupied: set[int] = set()
    for name, block in FEATURE_BLOCKS.items():
        indices = set(range(block.start or 0, block.stop or width, block.step or 1))
        if not indices or occupied & indices:
            raise ValueError(f"Invalid or overlapping feature block: {name}")
        occupied.update(indices)
    if occupied != set(range(width)):
        raise ValueError(f"Named feature blocks do not cover width {width}")


def modality_indices(subset: str, width: int = 3072) -> torch.Tensor:
    validate_feature_layout(width)
    if subset not in MODALITY_SUBSETS:
        raise ValueError(f"Unknown modality subset: {subset}")
    return torch.tensor([
        index
        for modality in MODALITY_SUBSETS[subset]
        for index in range(FEATURE_BLOCKS[modality].start, FEATURE_BLOCKS[modality].stop)
    ])


def select_modalities(features: torch.Tensor, subset: str) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != 3072:
        raise ValueError(f"Expected [rows, 3072] frozen features, got {tuple(features.shape)}")
    return features.index_select(1, modality_indices(subset, features.shape[1]))


def require_validation_selection(selection_split: str) -> None:
    if selection_split != "validation":
        raise ValueError("E4a model selection is validation-only; final test selection is forbidden")


def validation_selection_tensors(cache: dict, selection_split: str = "validation") -> tuple[torch.Tensor, torch.Tensor]:
    """Return only validation tensors even when a legacy cache also contains final-test tensors."""
    require_validation_selection(selection_split)
    return cache["val_features"], cache["val_labels"]


def _validation_manifest(base_cache: dict, base_cache_path: Path, missing: list[str]) -> dict:
    source = base_cache["manifest"]
    return {
        "schema_version": 1,
        "source_cache": str(base_cache_path.resolve()),
        "source_manifest": source,
        "conditions": missing,
        "selection_split": "validation",
        "test_rows_used": False,
    }


def prepare_missing_validation_features(
    base_cache: dict,
    base_cache_path: Path,
    validation_cache_path: Path,
    data_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict:
    present = set(base_cache.get("robust_val_conditions") or [])
    missing = [condition for condition in OFFICIAL_TRANSFORMED_CONDITIONS if condition not in present]
    manifest = _validation_manifest(base_cache, base_cache_path, missing)
    if validation_cache_path.exists():
        cached = torch.load(validation_cache_path, map_location="cpu", weights_only=True)
        if cached.get("manifest") != manifest:
            raise ValueError("Existing E4a validation cache is incompatible with the requested split/config")
        return cached
    if not missing:
        return {"features": torch.empty(0, 3072), "labels": torch.empty(0), "conditions": [], "manifest": manifest}

    from ..data import load_labeled_paths, stratified_train_val_test_split
    from ..features import extract_condition_features
    from ..model import FrozenEncoders, ModelConfig
    from ..train import choose_device

    source_manifest = base_cache["manifest"]
    config_values = source_manifest["model_config"]
    config = ModelConfig(**config_values)
    if config.clip_dim != 512 or config.forensic_dim != 2560 or config.forensic_mode != "laplacian_fft":
        raise ValueError("E4a requires the canonical 512+1280+1280 frozen feature layout")
    rows = load_labeled_paths(data_dir)
    _, val_rows, _ = stratified_train_val_test_split(
        rows, data_dir, source_manifest["validation_fraction"],
        source_manifest["test_fraction"], source_manifest["seed"],
    )
    expected_labels = torch.tensor([label for _, label in val_rows], dtype=torch.float32)
    if not torch.equal(base_cache["val_labels"], expected_labels):
        raise ValueError("Reconstructed validation split does not align with the base feature cache")
    device = choose_device(device_name)
    encoders = FrozenEncoders(config, device)
    features, labels, _, conditions = extract_condition_features(
        val_rows, encoders, batch_size, tuple(missing), source_manifest["seed"]
    )
    del encoders
    validation_cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": features, "labels": labels, "conditions": conditions, "manifest": manifest}
    torch.save(payload, validation_cache_path)
    return payload


def assemble_validation_matrix(base_cache: dict, extra_cache: dict) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    features = [base_cache["robust_val_features"]]
    labels = [base_cache["robust_val_labels"]]
    conditions = list(base_cache["robust_val_conditions"])
    if len(extra_cache["labels"]):
        features.append(extra_cache["features"])
        labels.append(extra_cache["labels"])
        conditions.extend(extra_cache["conditions"])
    if set(conditions) != set(OFFICIAL_TRANSFORMED_CONDITIONS):
        missing = sorted(set(OFFICIAL_TRANSFORMED_CONDITIONS) - set(conditions))
        raise ValueError(f"Validation matrix is incomplete; missing conditions: {missing}")
    return torch.cat(features), torch.cat(labels), conditions


def train_subset(
    subset: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    clean_x: torch.Tensor,
    clean_y: torch.Tensor,
    robust_x: torch.Tensor,
    robust_y: torch.Tensor,
    robust_conditions: list[str],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_x = select_modalities(train_x, subset)
    clean_x = select_modalities(clean_x, subset).to(device)
    robust_x = select_modalities(robust_x, subset).to(device)
    clean_y_device, robust_y_device = clean_y.to(device), robust_y.to(device)
    head = nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    best_state, best_loss, stale = None, float("inf"), 0
    condition_masks = {
        condition: torch.tensor([value == condition for value in robust_conditions], device=device)
        for condition in OFFICIAL_TRANSFORMED_CONDITIONS
    }
    for _ in range(epochs):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features.to(device)).squeeze(1), labels.to(device))
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            clean_loss = criterion(head(clean_x).squeeze(1), clean_y_device)
            robust_logits = head(robust_x).squeeze(1)
            condition_losses = torch.stack([
                criterion(robust_logits[condition_masks[name]], robust_y_device[condition_masks[name]])
                for name in OFFICIAL_TRANSFORMED_CONDITIONS
            ])
            selection_loss = 0.3 * clean_loss + 0.7 * 0.5 * (
                condition_losses.mean() + condition_losses.max()
            )
        if float(selection_loss) < best_loss - 1e-5:
            best_loss, stale = float(selection_loss), 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"No selected state for {subset}")
    head.load_state_dict(best_state)
    head.cpu().eval()
    with torch.no_grad():
        clean_logits = head(clean_x.cpu()).squeeze(1)
        robust_logits = head(robust_x.cpu()).squeeze(1)
    calibration_logits = torch.cat((clean_logits, robust_logits))
    calibration_labels = torch.cat((clean_y, robust_y))
    temperature = fit_temperature(calibration_logits, calibration_labels)
    calibration_probabilities = torch.sigmoid(calibration_logits / temperature)
    threshold = select_threshold(calibration_labels, calibration_probabilities, "balanced")
    clean_metrics = classification_metrics(clean_y, torch.sigmoid(clean_logits / temperature), threshold)
    per_condition = {}
    robust_probabilities = torch.sigmoid(robust_logits / temperature)
    for condition in OFFICIAL_TRANSFORMED_CONDITIONS:
        mask = torch.tensor([value == condition for value in robust_conditions])
        per_condition[condition] = classification_metrics(robust_y[mask], robust_probabilities[mask], threshold)
    baccs = [per_condition[name]["balanced_accuracy"] for name in OFFICIAL_TRANSFORMED_CONDITIONS]
    aucs = [per_condition[name]["roc_auc"] for name in OFFICIAL_TRANSFORMED_CONDITIONS]
    fprs = [per_condition[name]["false_positive_rate"] for name in OFFICIAL_TRANSFORMED_CONDITIONS]
    worst_index = int(np.argmin(baccs))
    return {
        "modality_subset": subset,
        "feature_dimension": int(train_x.shape[1]),
        "trainable_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "selection_split": "validation",
        "test_rows_used": False,
        "clean_validation_balanced_accuracy": clean_metrics["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
        "worst_condition": OFFICIAL_TRANSFORMED_CONDITIONS[worst_index],
        "per_condition_validation": per_condition,
        "mean_transformed_roc_auc": float(np.mean(aucs)),
        "worst_transformed_roc_auc": float(np.min(aucs)),
        "clean_false_positive_rate": clean_metrics["false_positive_rate"],
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
        "temperature": temperature,
        "threshold": threshold,
    }


def rank_ablation_rows(rows: list[dict], selection_split: str = "validation") -> list[dict]:
    require_validation_selection(selection_split)
    ranked = sorted(rows, key=lambda row: (
        row["worst_transformed_validation_balanced_accuracy"],
        row["mean_transformed_validation_balanced_accuracy"],
        row["clean_validation_balanced_accuracy"],
    ), reverse=True)
    return [{**row, "validation_rank": rank} for rank, row in enumerate(ranked, 1)]


def write_summaries(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_ablation_rows(rows)
    (output_dir / "ablation_summary.json").write_text(json.dumps({
        "selection_split": "validation", "test_rows_used": False,
        "ranking_policy": ["worst transformed balanced accuracy", "mean transformed balanced accuracy", "clean balanced accuracy"],
        "results": ranked,
    }, indent=2) + "\n", encoding="utf-8")
    scalar_keys = [key for key in ranked[0] if key != "per_condition_validation"]
    with (output_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys + ["per_condition_validation"])
        writer.writeheader()
        for row in ranked:
            writer.writerow({**row, "per_condition_validation": json.dumps(row["per_condition_validation"], sort_keys=True)})
    condition_rows = []
    for row in ranked:
        for condition, metrics in row["per_condition_validation"].items():
            condition_rows.append({"modality_subset": row["modality_subset"], "condition": condition, **metrics})
    with (output_dir / "modality_by_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)
    (output_dir / "modality_by_condition.json").write_text(
        json.dumps(condition_rows, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E4a frozen-modality robustness ablations")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require_validation_selection("validation")
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True, mmap=True)
    required = {"train_features", "train_labels", "val_features", "val_labels", "robust_val_features", "robust_val_labels", "robust_val_conditions", "manifest"}
    missing_keys = required - set(base)
    if missing_keys:
        raise ValueError(f"Base feature cache lacks required validation-only fields: {sorted(missing_keys)}")
    if "test_features" in base or "test_labels" in base:
        print("Ignoring test tensors present in base cache; E4a selection is validation-only.")
    extra = prepare_missing_validation_features(
        base, args.base_cache, args.validation_cache, args.data_dir, args.device, args.feature_batch_size
    )
    robust_x, robust_y, robust_conditions = assemble_validation_matrix(base, extra)
    from ..train import choose_device
    device = choose_device(args.device)
    clean_val_x, clean_val_y = validation_selection_tensors(base)
    rows = [train_subset(
        subset, base["train_features"], base["train_labels"], clean_val_x, clean_val_y,
        robust_x, robust_y, robust_conditions, args.epochs, args.patience, args.head_batch_size,
        args.learning_rate, args.seed, device,
    ) for subset in MODALITY_SUBSETS]
    write_summaries(rows, args.output_dir)
    print(f"E4a summaries written to: {args.output_dir}")


if __name__ == "__main__":
    main()
