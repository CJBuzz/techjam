from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_labeled_paths, stratified_train_val_test_split
from .metrics import classification_metrics, fit_temperature, select_threshold
from .features import extract_balanced_quality_statistics, extract_condition_quality_statistics
from .model import (
    AdaptiveTriExpertHead,
    ExpertMixtureHead,
    ModelConfig,
    image_quality_statistics,
    load_checkpoint,
    save_checkpoint,
)
from .train import _dataset_fingerprint, choose_device


def _copy_experts(
    head: ExpertMixtureHead | AdaptiveTriExpertHead, laplacian_path: Path, fused_path: Path
) -> None:
    laplacian_head, lap_config, _, _ = load_checkpoint(laplacian_path, torch.device("cpu"))
    fused_head, fused_config, _, _ = load_checkpoint(fused_path, torch.device("cpu"))
    if (
        lap_config.forensic_mode != "laplacian" or lap_config.head_type != "fusion"
        or fused_config.forensic_mode != "laplacian_fft" or fused_config.head_type != "fusion"
    ):
        raise ValueError("Expected FusionHead Laplacian and Laplacian+FFT initialization checkpoints")
    head.laplacian_expert.load_state_dict(laplacian_head.state_dict())
    if isinstance(head, AdaptiveTriExpertHead):
        source, target = laplacian_head.state_dict(), head.semantic_expert.state_dict()
        for key, value in target.items():
            if key == "network.0.weight":
                value.copy_(source[key][:, :512])
            else:
                value.copy_(source[key])
        head.semantic_expert.load_state_dict(target)
    source, target = fused_head.state_dict(), head.fft_expert.state_dict()
    for key, value in target.items():
        if key == "network.0.weight":
            value[:, :512].copy_(source[key][:, :512])
            value[:, 512:].copy_(source[key][:, 512 + 1280:])
        else:
            value.copy_(source[key])
    head.fft_expert.load_state_dict(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Laplacian/FFT expert mixture on cached features")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--laplacian-checkpoint", type=Path, required=True)
    parser.add_argument("--fused-checkpoint", type=Path, required=True)
    parser.add_argument("--gate-mode", choices=("fixed", "features", "quality"), default="features")
    parser.add_argument(
        "--experts", choices=("two", "three"), default="three",
        help="Three adds a corruption-resistant CLIP-only semantic expert",
    )
    parser.add_argument("--gate-prior-weight", type=float, default=0.01)
    parser.add_argument("--freeze-experts", action="store_true", help="Train only the learned gate")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--robust-validation-weight", type=float, default=0.7)
    parser.add_argument("--threshold-objective", choices=("balanced", "f1"), default="balanced")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.gate_prior_weight < 0:
        raise ValueError("--gate-prior-weight must be non-negative")
    if not 0.0 <= args.robust_validation_weight <= 1.0:
        raise ValueError("--robust-validation-weight must be in [0, 1]")
    if args.freeze_experts and args.gate_mode == "fixed":
        raise ValueError("A fixed gate with frozen experts has no trainable parameters")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = choose_device(args.device)
    rows = load_labeled_paths(args.data_dir)
    train_rows, val_rows, test_rows = stratified_train_val_test_split(
        rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
    )
    cached = torch.load(args.cache, map_location="cpu", weights_only=True)
    manifest = cached.get("manifest")
    if not manifest or cached.get("augmentation_policy") != "balanced":
        raise ValueError("Mixture training requires a current balanced feature cache with a manifest")
    if manifest.get("dataset_fingerprint") != _dataset_fingerprint(rows, args.data_dir):
        raise ValueError("Feature cache belongs to different dataset contents")
    if manifest.get("seed") != args.seed or manifest.get("validation_fraction") != args.validation_fraction \
            or manifest.get("test_fraction") != args.test_fraction:
        raise ValueError("Feature cache split settings do not match mixture training")
    train_x, train_y = cached["train_features"], cached["train_labels"]
    val_x, val_y = cached["val_features"], cached["val_labels"]
    robust_val_x, robust_val_y = cached.get("robust_val_features"), cached.get("robust_val_labels")
    robust_val_conditions = cached.get("robust_val_conditions")
    if robust_val_x is None or robust_val_y is None or not robust_val_conditions:
        raise ValueError("Mixture training requires robust validation features in the cache")
    if train_x.shape[1] != 3072 or val_x.shape[1] != 3072:
        raise ValueError("Mixture training requires a 3,072-wide Laplacian+FFT cache")
    quality_dim = 6 if args.gate_mode == "quality" else 0
    if quality_dim:
        repeats = len(train_y) // len(train_rows)
        train_x = torch.cat((train_x, extract_balanced_quality_statistics(train_rows, repeats, args.seed)), 1)
        val_stats = []
        from PIL import Image
        for path, _ in val_rows:
            with Image.open(path) as source: val_stats.append(image_quality_statistics([source.convert("RGB")])[0])
        val_x = torch.cat((val_x, torch.stack(val_stats)), 1)
        robust_stats = extract_condition_quality_statistics(
            val_rows, tuple(dict.fromkeys(robust_val_conditions)), args.seed
        )
        robust_val_x = torch.cat((robust_val_x, robust_stats), 1)
    config = ModelConfig(
        forensic_mode="laplacian_fft", forensic_dim=2560,
        head_type="tri_mixture" if args.experts == "three" else "mixture",
        gate_mode=args.gate_mode, quality_dim=quality_dim,
    )
    head = (
        AdaptiveTriExpertHead(config) if args.experts == "three" else ExpertMixtureHead(config)
    ).to(device)
    _copy_experts(head, args.laplacian_checkpoint, args.fused_checkpoint)
    if args.freeze_experts:
        head.laplacian_expert.requires_grad_(False)
        head.fft_expert.requires_grad_(False)
        if isinstance(head, AdaptiveTriExpertHead):
            head.semantic_expert.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.head_batch_size, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_state, best_loss, stale = None, float("inf"), 0
    val_x_device, val_y_device = val_x.to(device), val_y.to(device)
    robust_val_x_device, robust_val_y_device = robust_val_x.to(device), robust_val_y.to(device)
    for epoch in range(1, args.epochs + 1):
        head.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            if args.gate_mode in {"features", "quality"} and args.gate_prior_weight:
                gate = head.gate_weights(features)
                prior = (
                    torch.tensor((0.45, 0.35, 0.20), device=device)
                    if args.experts == "three" else torch.tensor(0.2, device=device)
                )
                loss = loss + args.gate_prior_weight * ((gate - prior) ** 2).mean()
            loss.backward(); optimizer.step()
        head.eval()
        with torch.no_grad():
            clean_loss = criterion(head(val_x_device), val_y_device)
            robust_logits = head(robust_val_x_device)
            condition_size = len(val_rows)
            condition_losses = torch.stack([
                criterion(
                    robust_logits[start : start + condition_size],
                    robust_val_y_device[start : start + condition_size],
                )
                for start in range(0, len(robust_logits), condition_size)
            ])
            robust_mean, robust_worst = condition_losses.mean(), condition_losses.max()
            selection_loss = (
                (1 - args.robust_validation_weight) * clean_loss
                + args.robust_validation_weight * 0.5 * (robust_mean + robust_worst)
            )
        print(
            f"epoch={epoch:03d} clean={float(clean_loss):.5f} robust_mean={float(robust_mean):.5f} "
            f"robust_worst={float(robust_worst):.5f} selection={float(selection_loss):.5f}"
        )
        if float(selection_loss) < best_loss - 1e-5:
            best_loss, stale = float(selection_loss), 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("Mixture training produced no checkpoint")
    head.load_state_dict(best_state); head.to("cpu").eval()
    with torch.no_grad():
        clean_logits = head(val_x)
        robust_logits = head(robust_val_x)
        calibration_logits = torch.cat((clean_logits, robust_logits))
        calibration_labels = torch.cat((val_y, robust_val_y))
        mean_gate = head.gate_weights(val_x).mean(0).tolist()
    temperature = fit_temperature(calibration_logits, calibration_labels)
    calibration_probabilities = torch.sigmoid(calibration_logits / temperature)
    threshold = select_threshold(calibration_labels, calibration_probabilities, args.threshold_objective)
    metrics = classification_metrics(val_y, torch.sigmoid(clean_logits / temperature), threshold)
    metadata = {
        "train_images": len(train_rows), "validation_images": len(val_rows), "test_images": len(test_rows),
        "seed": args.seed, "gate_mode": args.gate_mode, "gate_prior_weight": args.gate_prior_weight,
        "experts": args.experts, "robust_validation_weight": args.robust_validation_weight,
        "threshold": threshold, "threshold_objective": args.threshold_objective,
        "freeze_experts": args.freeze_experts,
        "mean_clean_validation_gate": mean_gate, "validation_metrics": metrics,
        "laplacian_checkpoint": str(args.laplacian_checkpoint), "fused_checkpoint": str(args.fused_checkpoint),
    }
    save_checkpoint(args.output, head, config, temperature, metadata)
    print(json.dumps({
        "checkpoint": str(args.output), "temperature": temperature, "threshold": threshold,
        "mean_gate": mean_gate, "validation": metrics,
    }, indent=2))


if __name__ == "__main__": main()
