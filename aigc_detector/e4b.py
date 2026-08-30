from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ROBUSTNESS_CONDITIONS
from .e4a import FEATURE_BLOCKS, assemble_validation_matrix, prepare_missing_validation_features
from .metrics import classification_metrics, fit_temperature, select_threshold
from .model import ModelConfig, load_checkpoint
from .train import choose_device


QUALITY_FEATURE_NAMES = (
    "clip_l2", "laplacian_l2", "fft_l2",
    "clip_mean_abs", "laplacian_mean_abs", "fft_mean_abs",
    "clip_std", "laplacian_std", "fft_std",
    "semantic_logit", "laplacian_logit", "fft_logit",
    "semantic_laplacian_disagreement", "semantic_fft_disagreement", "laplacian_fft_disagreement",
)
GATE_MODES = ("fixed", "global", "adaptive")
MODALITIES = ("semantic", "laplacian", "fft")


def split_modalities(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[1] != 3072:
        raise ValueError(f"Expected canonical [rows, 3072] features, got {tuple(features.shape)}")
    return tuple(features[:, block] for block in FEATURE_BLOCKS.values())  # type: ignore[return-value]


def quality_vector(features: torch.Tensor, modality_logits: torch.Tensor) -> torch.Tensor:
    """Deterministic inference-time reliability descriptors with no dataset metadata inputs."""
    if modality_logits.ndim != 2 or modality_logits.shape != (len(features), 3):
        raise ValueError("Expected three modality logits per feature row")
    blocks = split_modalities(features)
    l2 = [block.norm(dim=1, keepdim=True) for block in blocks]
    mean_abs = [block.abs().mean(dim=1, keepdim=True) for block in blocks]
    std = [block.std(dim=1, unbiased=False, keepdim=True) for block in blocks]
    disagreements = (
        (modality_logits[:, 0:1] - modality_logits[:, 1:2]).abs(),
        (modality_logits[:, 0:1] - modality_logits[:, 2:3]).abs(),
        (modality_logits[:, 1:2] - modality_logits[:, 2:3]).abs(),
    )
    result = torch.cat((*l2, *mean_abs, *std, modality_logits, *disagreements), dim=1)
    if result.shape[1] != len(QUALITY_FEATURE_NAMES):
        raise AssertionError("Unexpected E4b reliability-vector width")
    return result


class AdaptiveFusionHead(nn.Module):
    """Three linear experts with fixed, global, or feature-reliability gating."""

    def __init__(self, mode: str = "adaptive", gate_hidden_dim: int = 32) -> None:
        super().__init__()
        if mode not in GATE_MODES:
            raise ValueError(f"Unknown gate mode: {mode}")
        self.mode = mode
        self.gate_hidden_dim = gate_hidden_dim
        self.semantic_head = nn.Linear(512, 1)
        self.laplacian_head = nn.Linear(1280, 1)
        self.fft_head = nn.Linear(1280, 1)
        if mode == "fixed":
            self.register_buffer("fixed_weights", torch.full((3,), 1 / 3))
            self.gate = None
        elif mode == "global":
            self.global_logits = nn.Parameter(torch.zeros(3))
            self.gate = None
        else:
            self.gate = nn.Sequential(
                nn.Linear(len(QUALITY_FEATURE_NAMES), gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 3),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def modality_logits(self, features: torch.Tensor) -> torch.Tensor:
        clip, laplacian, fft = split_modalities(features)
        return torch.cat((
            self.semantic_head(clip), self.laplacian_head(laplacian), self.fft_head(fft)
        ), dim=1)

    def gate_weights(self, features: torch.Tensor, logits: torch.Tensor | None = None) -> torch.Tensor:
        if logits is None:
            logits = self.modality_logits(features)
        if self.mode == "fixed":
            return self.fixed_weights.expand(len(features), -1)
        if self.mode == "global":
            return torch.softmax(self.global_logits, dim=0).expand(len(features), -1)
        return torch.softmax(self.gate(quality_vector(features, logits)), dim=1)

    def forward(self, features: torch.Tensor, return_weights: bool = False):
        logits = self.modality_logits(features)
        weights = self.gate_weights(features, logits)
        fused = (weights * logits).sum(dim=1)
        return (fused, weights) if return_weights else fused


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def require_validation_selection(split: str) -> None:
    if split != "validation":
        raise ValueError("E4b model selection is validation-only; final-test selection is forbidden")


def aggregate_gate_statistics(
    weights: torch.Tensor, labels: torch.Tensor, conditions: list[str]
) -> dict[str, dict[str, list[float] | int]]:
    if weights.shape != (len(labels), 3) or len(conditions) != len(labels):
        raise ValueError("Gate weights, labels, and conditions must align")
    result = {}
    for condition in sorted(set(conditions), key=conditions.index):
        condition_mask = torch.tensor([value == condition for value in conditions])
        groups = {"overall": condition_mask, "real": condition_mask & (labels == 0), "fake": condition_mask & (labels == 1)}
        for group, mask in groups.items():
            if mask.any():
                result[f"{condition}:{group}"] = {
                    "count": int(mask.sum()), "weights": weights[mask].mean(0).tolist()
                }
    return result


def save_adaptive_checkpoint(
    path: Path, model: AdaptiveFusionHead, temperature: float, threshold: float, metadata: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "state_dict": model.state_dict(), "mode": model.mode,
        "gate_hidden_dim": model.gate_hidden_dim, "temperature": temperature,
        "threshold": threshold, "metadata": metadata,
    }, temporary)
    os.replace(temporary, path)


def load_adaptive_checkpoint(path: Path, device: torch.device) -> tuple[AdaptiveFusionHead, float, float, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = AdaptiveFusionHead(payload["mode"], payload["gate_hidden_dim"]).to(device)
    result = model.load_state_dict(payload["state_dict"])
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("Incompatible E4b checkpoint state")
    model.eval()
    return model, float(payload["temperature"]), float(payload["threshold"]), payload["metadata"]


def _score(
    model: AdaptiveFusionHead, clean_x: torch.Tensor, clean_y: torch.Tensor,
    robust_x: torch.Tensor, robust_y: torch.Tensor, robust_conditions: list[str],
    temperature: float, threshold: float,
) -> tuple[dict, dict, dict]:
    model.cpu().eval()
    with torch.no_grad():
        clean_logits, clean_weights = model(clean_x, return_weights=True)
        robust_logits, robust_weights = model(robust_x, return_weights=True)
    clean_metrics = classification_metrics(clean_y, torch.sigmoid(clean_logits / temperature), threshold)
    robust_probabilities = torch.sigmoid(robust_logits / temperature)
    per_condition = {}
    for condition in ROBUSTNESS_CONDITIONS[1:]:
        mask = torch.tensor([value == condition for value in robust_conditions])
        per_condition[condition] = classification_metrics(
            robust_y[mask], robust_probabilities[mask], threshold
        )
    labels = torch.cat((clean_y, robust_y))
    weights = torch.cat((clean_weights, robust_weights))
    conditions = ["clean"] * len(clean_y) + robust_conditions
    gate_stats = aggregate_gate_statistics(weights, labels, conditions)
    return clean_metrics, per_condition, gate_stats


def _summary(
    mode: str, model: AdaptiveFusionHead, clean_metrics: dict, per_condition: dict,
    gate_stats: dict, baseline: dict,
) -> dict:
    conditions = list(ROBUSTNESS_CONDITIONS[1:])
    baccs = [per_condition[name]["balanced_accuracy"] for name in conditions]
    aucs = [per_condition[name]["roc_auc"] for name in conditions]
    fprs = [per_condition[name]["false_positive_rate"] for name in conditions]
    worst_index = int(np.argmin(baccs))
    return {
        "mode": mode, "trainable_parameter_count": parameter_count(model),
        "selection_split": "validation", "test_rows_used_for_selection": False,
        "clean_validation_balanced_accuracy": clean_metrics["balanced_accuracy"],
        "mean_transformed_validation_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_validation_balanced_accuracy": float(baccs[worst_index]),
        "worst_condition": conditions[worst_index],
        "mean_transformed_roc_auc": float(np.mean(aucs)),
        "worst_transformed_roc_auc": float(np.min(aucs)),
        "clean_false_positive_rate": clean_metrics["false_positive_rate"],
        "mean_transformed_false_positive_rate": float(np.mean(fprs)),
        "baseline_comparison": {
            "checkpoint": baseline["checkpoint"],
            "clean_balanced_accuracy_delta": clean_metrics["balanced_accuracy"] - baseline["clean_balanced_accuracy"],
            "mean_transformed_balanced_accuracy_delta": float(np.mean(baccs)) - baseline["mean_transformed_balanced_accuracy"],
            "worst_transformed_balanced_accuracy_delta": float(np.min(baccs)) - baseline["worst_transformed_balanced_accuracy"],
        },
        "average_gate_weights_by_condition": gate_stats,
    }


def _baseline_metrics(
    checkpoint: Path, clean_x: torch.Tensor, clean_y: torch.Tensor,
    robust_x: torch.Tensor, robust_y: torch.Tensor, conditions: list[str],
) -> dict:
    head, config, temperature, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    with torch.no_grad():
        clean_p = torch.sigmoid(head(clean_x) / temperature)
        robust_p = torch.sigmoid(head(robust_x) / temperature)
    threshold = float(metadata.get("threshold", 0.5))
    clean = classification_metrics(clean_y, clean_p, threshold)
    baccs = []
    for condition in ROBUSTNESS_CONDITIONS[1:]:
        mask = torch.tensor([value == condition for value in conditions])
        baccs.append(classification_metrics(robust_y[mask], robust_p[mask], threshold)["balanced_accuracy"])
    return {
        "checkpoint": str(checkpoint.resolve()), "clean_balanced_accuracy": clean["balanced_accuracy"],
        "mean_transformed_balanced_accuracy": float(np.mean(baccs)),
        "worst_transformed_balanced_accuracy": float(np.min(baccs)),
    }


def train_command(args: argparse.Namespace) -> None:
    require_validation_selection("validation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True, mmap=True)
    if base.get("augmentation_policy") != "balanced":
        raise ValueError("E4b requires the existing balanced local feature cache")
    extra = prepare_missing_validation_features(
        base, args.base_cache, args.validation_cache, args.data_dir, args.device, args.feature_batch_size
    )
    robust_x, robust_y, robust_conditions = assemble_validation_matrix(base, extra)
    train_x, train_y = base["train_features"], base["train_labels"]
    train_groups = base["train_groups"]
    original_indices = base["train_original_indices"]
    originals = int(original_indices.max()) + 1
    repeats = len(train_y) // originals
    if repeats * originals != len(train_y):
        raise ValueError("Incomplete local robust-view pairing")
    device = choose_device(args.device)
    model = AdaptiveFusionHead(args.mode, args.gate_hidden_dim).to(device)
    if parameter_count(model) >= 100_000:
        raise ValueError("E4b must remain below 100k trainable parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loader = DataLoader(torch.arange(originals), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    group_names = sorted(set(train_groups))
    clean_x, clean_y = base["val_features"], base["val_labels"]
    baseline = _baseline_metrics(
        args.baseline_checkpoint, clean_x, clean_y, robust_x, robust_y, robust_conditions
    )
    clean_floor = baseline["clean_balanced_accuracy"] - 0.01
    best_state, best_key, stale = None, None, 0
    for _ in range(args.epochs):
        model.train()
        for original_batch in loader:
            indices = torch.cat([original_batch + repeat * originals for repeat in range(repeats)])
            features, labels = train_x[indices].to(device), train_y[indices].to(device)
            groups = [train_groups[index] for index in indices.tolist()]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = losses.mean() + 0.05 * ((logits.view(repeats, -1) - logits.view(repeats, -1).mean(0)) ** 2).mean()
            group_losses = [
                losses[torch.tensor([value == group for value in groups], device=device)].mean()
                for group in group_names if group in groups
            ]
            loss = loss + 0.5 * torch.stack(group_losses).max()
            loss.backward()
            optimizer.step()
        model.cpu().eval()
        with torch.no_grad():
            clean_logits = model(clean_x)
            robust_logits = model(robust_x)
        threshold = select_threshold(
            torch.cat((clean_y, robust_y)), torch.sigmoid(torch.cat((clean_logits, robust_logits))), "balanced"
        )
        clean_metric = classification_metrics(clean_y, torch.sigmoid(clean_logits), threshold)
        condition_baccs = []
        for condition in ROBUSTNESS_CONDITIONS[1:]:
            mask = torch.tensor([value == condition for value in robust_conditions])
            condition_baccs.append(classification_metrics(
                robust_y[mask], torch.sigmoid(robust_logits[mask]), threshold
            )["balanced_accuracy"])
        key = (clean_metric["balanced_accuracy"] >= clean_floor, min(condition_baccs), float(np.mean(condition_baccs)))
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
        model.to(device)
    if best_state is None:
        raise RuntimeError("E4b training produced no validation-selected state")
    model.load_state_dict(best_state)
    model.cpu().eval()
    with torch.no_grad():
        clean_logits = model(clean_x)
        robust_logits = model(robust_x)
    calibration_logits = torch.cat((clean_logits, robust_logits))
    calibration_labels = torch.cat((clean_y, robust_y))
    temperature = fit_temperature(calibration_logits, calibration_labels)
    threshold = select_threshold(
        calibration_labels, torch.sigmoid(calibration_logits / temperature), "balanced"
    )
    clean_metrics, per_condition, gate_stats = _score(
        model, clean_x, clean_y, robust_x, robust_y, robust_conditions, temperature, threshold
    )
    summary = _summary(args.mode, model, clean_metrics, per_condition, gate_stats, baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "E4b", "mode": args.mode, "selection_split": "validation",
        "test_rows_used_for_selection": False, "base_cache_manifest": base["manifest"],
        "validation_cache_manifest": extra["manifest"], "baseline": baseline,
        "summary": summary,
    }
    save_adaptive_checkpoint(args.output_dir / "model.pt", model, temperature, threshold, metadata)
    config = {
        "mode": args.mode, "gate_hidden_dim": args.gate_hidden_dim,
        "quality_feature_names": list(QUALITY_FEATURE_NAMES),
        "feature_blocks": {name: [block.start, block.stop] for name, block in FEATURE_BLOCKS.items()},
        "equal_gate_initialization": True, "seed": args.seed,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    scorecard = {"clean": clean_metrics, **per_condition}
    (args.output_dir / "validation_scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "validation_per_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", *clean_metrics])
        writer.writeheader()
        for condition, metrics in scorecard.items():
            writer.writerow({"condition": condition, **metrics})
    with (args.output_dir / "gate_weights_by_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "label_group", "count", "semantic_weight", "laplacian_weight", "fft_weight"])
        writer.writeheader()
        for key, values in gate_stats.items():
            condition, group = key.split(":")
            writer.writerow({"condition": condition, "label_group": group, "count": values["count"],
                             **dict(zip(("semantic_weight", "laplacian_weight", "fft_weight"), values["weights"], strict=True))})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def locked_test_command(args: argparse.Namespace) -> None:
    model, temperature, threshold, metadata = load_adaptive_checkpoint(args.model, torch.device("cpu"))
    if metadata.get("selection_split") != "validation" or metadata.get("test_rows_used_for_selection") is not False:
        raise ValueError("E4b checkpoint is not a validation-locked candidate")
    from .data import load_labeled_paths, stratified_train_val_test_split
    from .features import extract_condition_features, extract_features
    from .model import FrozenEncoders
    rows = load_labeled_paths(args.data_dir)
    _, _, test_rows = stratified_train_val_test_split(rows, args.data_dir, 0.15, 0.15, args.seed)
    device = choose_device(args.device)
    config = ModelConfig(forensic_mode="laplacian_fft", forensic_dim=2560)
    encoders = FrozenEncoders(config, device)
    clean_x, clean_y, _ = extract_features(test_rows, encoders, args.feature_batch_size)
    robust_x, robust_y, _, conditions = extract_condition_features(
        test_rows, encoders, args.feature_batch_size, ROBUSTNESS_CONDITIONS[1:], args.seed
    )
    del encoders
    clean, per_condition, gates = _score(
        model, clean_x, clean_y, robust_x, robust_y, conditions, temperature, threshold
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_test_scorecard.json").write_text(json.dumps({
        "evaluation_split": "test", "model_selection_performed": False,
        "clean": clean, "per_condition": per_condition, "gate_weights": gates,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="E4b lightweight quality-aware adaptive fusion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--base-cache", type=Path, required=True)
    train.add_argument("--validation-cache", type=Path, required=True)
    train.add_argument("--baseline-checkpoint", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--mode", choices=GATE_MODES, default="adaptive")
    train.add_argument("--gate-hidden-dim", type=int, default=32)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--patience", type=int, default=6)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--feature-batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    train.set_defaults(handler=train_command)
    test = subparsers.add_parser("locked-test")
    test.add_argument("--data-dir", type=Path, required=True)
    test.add_argument("--model", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--feature-batch-size", type=int, default=8)
    test.add_argument("--seed", type=int, default=42)
    test.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    test.set_defaults(handler=locked_test_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
