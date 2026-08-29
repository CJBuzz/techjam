from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_labeled_paths, stratified_train_val_test_split
from .metrics import classification_metrics, fit_temperature
from .features import extract_balanced_quality_statistics
from .model import ExpertMixtureHead, ModelConfig, image_quality_statistics, load_checkpoint, save_checkpoint
from .train import choose_device


def _copy_experts(head: ExpertMixtureHead, laplacian_path: Path, fused_path: Path) -> None:
    laplacian_head, lap_config, _, _ = load_checkpoint(laplacian_path, torch.device("cpu"))
    fused_head, fused_config, _, _ = load_checkpoint(fused_path, torch.device("cpu"))
    if lap_config.forensic_mode != "laplacian" or fused_config.forensic_mode != "laplacian_fft":
        raise ValueError("Expected Laplacian and Laplacian+FFT initialization checkpoints")
    head.laplacian_expert.load_state_dict(laplacian_head.state_dict())
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
    parser.add_argument("--gate-prior-weight", type=float, default=0.01)
    parser.add_argument("--freeze-experts", action="store_true", help="Train only the learned gate")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = choose_device(args.device)
    rows = load_labeled_paths(args.data_dir)
    train_rows, val_rows, test_rows = stratified_train_val_test_split(
        rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
    )
    cached = torch.load(args.cache, map_location="cpu", weights_only=True)
    train_x, train_y = cached["train_features"], cached["train_labels"]
    val_x, val_y = cached["val_features"], cached["val_labels"]
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
    config = ModelConfig(
        forensic_mode="laplacian_fft", forensic_dim=2560, head_type="mixture",
        gate_mode=args.gate_mode, quality_dim=quality_dim,
    )
    head = ExpertMixtureHead(config).to(device)
    _copy_experts(head, args.laplacian_checkpoint, args.fused_checkpoint)
    if args.freeze_experts:
        head.laplacian_expert.requires_grad_(False)
        head.fft_expert.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.head_batch_size, shuffle=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_state, best_loss, stale = None, float("inf"), 0
    val_x_device, val_y_device = val_x.to(device), val_y.to(device)
    for epoch in range(1, args.epochs + 1):
        head.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            if args.gate_mode in {"features", "quality"} and args.gate_prior_weight:
                gate = head.gate_weights(features)
                loss = loss + args.gate_prior_weight * ((gate - 0.2) ** 2).mean()
            loss.backward(); optimizer.step()
        head.eval()
        with torch.no_grad(): val_loss = float(criterion(head(val_x_device), val_y_device))
        print(f"epoch={epoch:03d} val_loss={val_loss:.5f}")
        if val_loss < best_loss - 1e-5:
            best_loss, stale = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("Mixture training produced no checkpoint")
    head.load_state_dict(best_state); head.to("cpu").eval()
    with torch.no_grad():
        logits = head(val_x)
        mean_gate = float(head.gate_weights(val_x).mean())
    temperature = fit_temperature(logits, val_y)
    metrics = classification_metrics(val_y, torch.sigmoid(logits / temperature))
    metadata = {
        "train_images": len(train_rows), "validation_images": len(val_rows), "test_images": len(test_rows),
        "seed": args.seed, "gate_mode": args.gate_mode, "gate_prior_weight": args.gate_prior_weight,
        "freeze_experts": args.freeze_experts,
        "mean_clean_validation_fft_gate": mean_gate, "validation_metrics": metrics,
        "laplacian_checkpoint": str(args.laplacian_checkpoint), "fused_checkpoint": str(args.fused_checkpoint),
    }
    save_checkpoint(args.output, head, config, temperature, metadata)
    print(json.dumps({"checkpoint": str(args.output), "temperature": temperature, "mean_fft_gate": mean_gate, "validation": metrics}, indent=2))


if __name__ == "__main__": main()
