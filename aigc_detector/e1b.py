from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ROBUST_SELECTION_CONDITIONS
from .metrics import classification_metrics, fit_temperature, select_threshold
from .model import FusionHead, ModelConfig, load_checkpoint, save_checkpoint
from .streaming_cache import load_stream_feature_cache
from .train import choose_device


DEFAULT_WEIGHTS = (0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
WEIGHT_DEFINITION = (
    "For per-view losses l_i, local examples have source weight 1 and external examples "
    "have source weight w: L=(sum_local(l_i)+w*sum_external(l_i))/(N_local+w*N_external). "
    "The same weights normalize paired consistency and within-group losses."
)


def weighted_mean(losses: torch.Tensor, external_mask: torch.Tensor, external_weight: float) -> torch.Tensor:
    if not 0.0 <= external_weight <= 1.0:
        raise ValueError("external_weight must be in [0, 1]")
    weights = torch.where(external_mask, external_weight, 1.0).to(losses)
    if not bool((weights > 0).any()):
        raise ValueError("At least one local loss is required")
    return (losses * weights).sum() / weights.sum()


@dataclass
class PairedFeatureStore:
    local_x: torch.Tensor
    local_y: torch.Tensor
    local_groups: list[str]
    local_originals: int
    external_x: torch.Tensor
    external_y: torch.Tensor
    external_groups: list[str]
    external_originals: int

    def __post_init__(self) -> None:
        self.repeats = len(self.local_y) // self.local_originals
        if self.repeats * self.local_originals != len(self.local_y):
            raise ValueError("Local cache does not contain complete paired repeats")
        external_repeats = len(self.external_y) // self.external_originals
        if external_repeats != self.repeats or external_repeats * self.external_originals != len(self.external_y):
            raise ValueError("External cache repeat structure does not match local cache")
        if len(self.local_groups) != len(self.local_y) or len(self.external_groups) != len(self.external_y):
            raise ValueError("Group metadata does not align with cached features")
        if self.local_x.shape[1] != self.external_x.shape[1]:
            raise ValueError("Local/external feature dimensions differ")

    @property
    def total_originals(self) -> int:
        return self.local_originals + self.external_originals

    def batch(self, original_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        """Gather repeat-major paired views without concatenating full cache tensors."""
        indices = original_indices.to(dtype=torch.long, device="cpu")
        is_external_original = indices >= self.local_originals
        feature_repeats, label_repeats, group_repeats, source_repeats = [], [], [], []
        for repeat in range(self.repeats):
            features = torch.empty(len(indices), self.local_x.shape[1], dtype=self.local_x.dtype)
            labels = torch.empty(len(indices), dtype=self.local_y.dtype)
            groups = [""] * len(indices)
            local_positions = (~is_external_original).nonzero().flatten()
            external_positions = is_external_original.nonzero().flatten()
            if len(local_positions):
                local_indices = indices[local_positions]
                features[local_positions] = self.local_x[repeat * self.local_originals + local_indices]
                labels[local_positions] = self.local_y[repeat * self.local_originals + local_indices]
                for position, index in zip(local_positions.tolist(), local_indices.tolist(), strict=True):
                    groups[position] = self.local_groups[repeat * self.local_originals + index]
            if len(external_positions):
                external_indices = indices[external_positions] - self.local_originals
                features[external_positions] = self.external_x[
                    repeat * self.external_originals + external_indices
                ]
                labels[external_positions] = self.external_y[
                    repeat * self.external_originals + external_indices
                ]
                for position, index in zip(external_positions.tolist(), external_indices.tolist(), strict=True):
                    groups[position] = self.external_groups[repeat * self.external_originals + index]
            feature_repeats.append(features)
            label_repeats.append(labels)
            group_repeats.extend(groups)
            source_repeats.append(is_external_original.clone())
        return (
            torch.cat(feature_repeats), torch.cat(label_repeats), group_repeats,
            torch.cat(source_repeats),
        )


def require_validation_selection(split: str) -> None:
    if split != "validation":
        raise ValueError("E1b selection is validation-only; final test selection is forbidden")


def contribution_statistics(local_originals: int, external_originals: int, repeats: int, weight: float) -> dict:
    local_rows, external_rows = local_originals * repeats, external_originals * repeats
    local_mass, external_mass = float(local_rows), float(weight * external_rows)
    total_mass = local_mass + external_mass
    return {
        "local_originals": local_originals,
        "external_originals_available": external_originals,
        "paired_views_per_original": repeats,
        "local_feature_rows": local_rows,
        "external_feature_rows_available": external_rows,
        "local_weight_mass": local_mass,
        "external_weight_mass": external_mass,
        "external_effective_fraction": external_mass / total_mass,
        "external_weight_definition": WEIGHT_DEFINITION,
    }


def rank_results(rows: list[dict], selection_split: str = "validation") -> list[dict]:
    require_validation_selection(selection_split)
    succeeded = [row for row in rows if row.get("status") == "succeeded"]
    local = next((row for row in succeeded if float(row["external_weight"]) == 0.0), None)
    if local is None:
        return [{**row, "clean_constraint_passed": None, "primary_track5_rank": None} for row in rows]
    floor = local["clean_validation_balanced_accuracy"] - 0.01
    eligible = [row for row in succeeded if row["clean_validation_balanced_accuracy"] >= floor]
    eligible.sort(key=lambda row: (
        row["worst_transformed_validation_balanced_accuracy"],
        row["mean_transformed_validation_balanced_accuracy"],
    ), reverse=True)
    ranks = {float(row["external_weight"]): rank for rank, row in enumerate(eligible, 1)}
    return [{
        **row,
        "clean_constraint_passed": (
            row["clean_validation_balanced_accuracy"] >= floor
            if row.get("status") == "succeeded" else None
        ),
        "primary_track5_rank": ranks.get(float(row["external_weight"])),
    } for row in rows]


def weight_name(weight: float) -> str:
    return f"weight_{weight:.2f}".replace(".", "p")


def completed_result(directory: Path, weight: float) -> dict | None:
    result_path, checkpoint = directory / "result.json", directory / "model.pt"
    if not result_path.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "succeeded" or float(result.get("external_weight", -1)) != weight:
        return None
    if result.get("selection_split") != "validation" or result.get("test_rows_used") is not False:
        raise ValueError(f"Unsafe completed E1b result metadata: {result_path}")
    return result


def train_weight(
    weight: float, store: PairedFeatureStore, local_cache: dict, initial_checkpoint: Path,
    output_dir: Path, device: torch.device, epochs: int, patience: int, batch_size: int,
    learning_rate: float, seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    initial_head, config, _, _ = load_checkpoint(initial_checkpoint, torch.device("cpu"))
    if asdict(config) != local_cache["manifest"]["model_config"]:
        raise ValueError("Initialization checkpoint and frozen feature cache configurations differ")
    head = FusionHead(config).to(device)
    head.load_state_dict(initial_head.state_dict())
    del initial_head
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    total_originals = store.local_originals if weight == 0 else store.total_originals
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(torch.arange(total_originals), batch_size=batch_size, shuffle=True, generator=generator)
    group_names = sorted(set(store.local_groups + (store.external_groups if weight else [])))
    val_x, val_y = local_cache["val_features"].to(device), local_cache["val_labels"].to(device)
    robust_x = local_cache["robust_val_features"].to(device)
    robust_y = local_cache["robust_val_labels"].to(device)
    robust_conditions = local_cache["robust_val_conditions"]
    condition_masks = {
        condition: torch.tensor([value == condition for value in robust_conditions], device=device)
        for condition in ROBUST_SELECTION_CONDITIONS
    }
    best_state, best_loss, stale = None, float("inf"), 0
    for _ in range(epochs):
        head.train()
        for originals in loader:
            features, labels, groups, external_mask = store.batch(originals)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features.to(device))
            losses = F.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="none")
            external_mask = external_mask.to(device)
            loss = weighted_mean(losses, external_mask, weight)
            paired_logits = logits.view(store.repeats, -1)
            consistency = ((paired_logits - paired_logits.mean(0)) ** 2).mean(0)
            loss = loss + 0.05 * weighted_mean(
                consistency, external_mask[: len(originals)], weight
            )
            group_losses = []
            for group in group_names:
                mask = torch.tensor([value == group for value in groups], device=device)
                if mask.any() and bool(((~external_mask[mask]) | (weight > 0)).any()):
                    group_losses.append(weighted_mean(losses[mask], external_mask[mask], weight))
            loss = loss + 0.5 * torch.stack(group_losses).max()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            clean_loss = F.binary_cross_entropy_with_logits(head(val_x), val_y)
            robust_logits = head(robust_x)
            condition_losses = torch.stack([
                F.binary_cross_entropy_with_logits(
                    robust_logits[condition_masks[name]], robust_y[condition_masks[name]]
                ) for name in ROBUST_SELECTION_CONDITIONS
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
        raise RuntimeError("E1b training produced no selected state")
    head.load_state_dict(best_state)
    head.cpu().eval()
    with torch.no_grad():
        clean_logits = head(val_x.cpu())
        robust_logits = head(robust_x.cpu())
    calibration_logits = torch.cat((clean_logits, robust_logits))
    calibration_labels = torch.cat((val_y.cpu(), robust_y.cpu()))
    temperature = fit_temperature(calibration_logits, calibration_labels)
    threshold = select_threshold(
        calibration_labels, torch.sigmoid(calibration_logits / temperature), "balanced"
    )
    clean_metrics = classification_metrics(
        val_y.cpu(), torch.sigmoid(clean_logits / temperature), threshold
    )
    robust_probabilities = torch.sigmoid(robust_logits / temperature)
    per_condition = {}
    for condition in ROBUST_SELECTION_CONDITIONS:
        mask = torch.tensor([value == condition for value in robust_conditions])
        per_condition[condition] = classification_metrics(
            robust_y.cpu()[mask], robust_probabilities[mask], threshold
        )
    baccs = [per_condition[name]["balanced_accuracy"] for name in ROBUST_SELECTION_CONDITIONS]
    aucs = [per_condition[name]["roc_auc"] for name in ROBUST_SELECTION_CONDITIONS]
    fprs = [per_condition[name]["false_positive_rate"] for name in ROBUST_SELECTION_CONDITIONS]
    worst_index = int(np.argmin(baccs))
    checkpoint = output_dir / "model.pt"
    stats = contribution_statistics(
        store.local_originals, store.external_originals, store.repeats, weight
    )
    metadata = {
        "experiment": "E1b", "external_weight": weight,
        "external_weight_definition": WEIGHT_DEFINITION, "training_contribution": stats,
        "selection_split": "validation", "test_rows_used": False,
        "threshold": threshold, "validation_metrics": clean_metrics,
        "robust_validation_metrics": per_condition,
    }
    temporary = checkpoint.with_suffix(".pt.tmp")
    save_checkpoint(temporary, head, config, temperature, metadata)
    os.replace(temporary, checkpoint)
    return {
        "external_weight": weight, "checkpoint": str(checkpoint), "status": "succeeded",
        "selection_split": "validation", "test_rows_used": False,
        "clean_validation_balanced_accuracy": clean_metrics["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
        "worst_condition": ROBUST_SELECTION_CONDITIONS[worst_index],
        "clean_false_positive_rate": clean_metrics["false_positive_rate"],
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
        "mean_transformed_roc_auc": float(np.mean(aucs)),
        "worst_transformed_roc_auc": float(np.min(aucs)),
        "training_contribution": stats, "per_condition_validation": per_condition,
        "external_roc_auc": None, "external_balanced_accuracy": None,
        "external_precision": None, "external_recall": None, "external_false_positive_rate": None,
    }


def write_summary(rows: list[dict], output_dir: Path) -> None:
    ranked = rank_results(rows)
    document = {
        "selection_split": "validation", "test_rows_used": False,
        "weight_definition": WEIGHT_DEFINITION,
        "ranking_policy": {
            "primary": "worst transformed validation balanced accuracy",
            "constraint": "clean validation BAcc >= weight-0 clean validation BAcc - 0.01",
            "tie_break": "mean transformed validation balanced accuracy",
            "external_metrics_used": False,
        },
        "results": ranked,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "weight_sweep_summary.json"
    csv_path = output_dir / "weight_sweep_summary.csv"
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    scalar_keys = sorted({
        key for row in ranked for key, value in row.items()
        if not isinstance(value, dict) and key != "training_contribution"
    })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys + ["training_contribution"])
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row.get(key) for key in scalar_keys} | {
                "training_contribution": json.dumps(row.get("training_contribution"), sort_keys=True)
            })


def load_caches(local_path: Path, external_path: Path) -> tuple[dict, PairedFeatureStore]:
    local = torch.load(local_path, map_location="cpu", weights_only=True, mmap=True)
    if local.get("augmentation_policy") != "balanced":
        raise ValueError("E1b requires the existing balanced local cache")
    required = {"train_features", "train_labels", "train_groups", "train_original_indices",
                "val_features", "val_labels", "robust_val_features", "robust_val_labels",
                "robust_val_conditions", "manifest"}
    if required - set(local):
        raise ValueError(f"Local cache missing keys: {sorted(required - set(local))}")
    if "test_features" in local or "test_labels" in local:
        print("Ignoring final-test tensors in local cache; E1b selection is validation-only.")
    local_originals = int(local["train_original_indices"].max()) + 1
    external_x, external_y, external_groups, _, external_originals, _ = load_stream_feature_cache(
        external_path, local["manifest"]["model_config"]
    )
    store = PairedFeatureStore(
        local["train_features"], local["train_labels"], local["train_groups"], local_originals,
        external_x, external_y, external_groups, external_originals,
    )
    return local, store


def sweep_command(args: argparse.Namespace) -> None:
    require_validation_selection("validation")
    local, store = load_caches(args.local_cache, args.external_cache)
    device = choose_device(args.device)
    rows = []
    for weight in args.weights:
        directory = args.output_dir / weight_name(weight)
        existing = completed_result(directory, weight)
        if existing:
            print(f"Skipping completed weight={weight:.2f}: {directory}")
            rows.append(existing)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        try:
            result = train_weight(
                weight, store, local, args.initialize_from_checkpoint, directory, device,
                args.epochs, args.patience, args.batch_size, args.learning_rate, args.seed,
            )
        except Exception as error:
            result = {
                "external_weight": weight, "checkpoint": str(directory / "model.pt"),
                "status": "failed", "failure_reason": f"{type(error).__name__}: {error}",
                "selection_split": "validation", "test_rows_used": False,
            }
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append(result)
        write_summary(rows, args.output_dir)
    write_summary(rows, args.output_dir)


def external_command(args: argparse.Namespace) -> None:
    summary_path = args.output_dir / "weight_sweep_summary.json"
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = document["results"]
    for row in rows:
        if row.get("status") != "succeeded":
            continue
        directory = args.output_dir / weight_name(float(row["external_weight"])) / "external"
        external_summary = directory / "summary.json"
        if not external_summary.exists():
            subprocess.run([
                sys.executable, "-m", "aigc_detector.evaluate_external",
                "--data-dir", str(args.data_dir), "--checkpoint", row["checkpoint"],
                "--output-dir", str(directory), "--batch-size", str(args.batch_size),
                "--device", args.device,
            ], check=True)
        payload = json.loads(external_summary.read_text(encoding="utf-8"))["overall"]
        with (directory / "predictions.csv").open(encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
        negatives = [item for item in predictions if int(item["true_label"]) == 0]
        false_positives = sum(int(item["predicted_label"]) == 1 for item in negatives)
        row.update({
            "external_roc_auc": payload["roc_auc"],
            "external_balanced_accuracy": payload["balanced_accuracy"],
            "external_precision": payload["precision"], "external_recall": payload["recall"],
            "external_false_positive_rate": false_positives / len(negatives),
        })
        result_path = args.output_dir / weight_name(float(row["external_weight"])) / "result.json"
        result_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    write_summary(rows, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1b external-contribution weight sweep")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--local-cache", type=Path, required=True)
    sweep.add_argument("--external-cache", type=Path, required=True)
    sweep.add_argument("--initialize-from-checkpoint", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, required=True)
    sweep.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    sweep.add_argument("--epochs", type=int, default=30)
    sweep.add_argument("--patience", type=int, default=6)
    sweep.add_argument("--batch-size", type=int, default=32)
    sweep.add_argument("--learning-rate", type=float, default=1e-3)
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    sweep.set_defaults(handler=sweep_command)
    external = subparsers.add_parser("external")
    external.add_argument("--data-dir", type=Path, required=True)
    external.add_argument("--output-dir", type=Path, required=True)
    external.add_argument("--batch-size", type=int, default=8)
    external.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    external.set_defaults(handler=external_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
