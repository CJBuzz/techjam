from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_labeled_paths, stratified_train_val_test_split
from .features import extract_features
from .metrics import classification_metrics, fit_temperature
from .model import FrozenEncoders, FusionHead, ModelConfig, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen two-stream AIGC detector")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing real/ and ai/ folders")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hybrid_detector.pt"))
    parser.add_argument("--cache", type=Path, default=None, help="Optional feature cache (.pt)")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--report-test-metrics",
        action="store_true",
        help="Score the test split during training (omit during model selection; use aigc-evaluate for the final model)",
    )
    parser.add_argument("--augmentation-repeats", type=int, default=2, help="First pass is clean; remaining passes are robust")
    parser.add_argument("--augmentation-depth", type=int, default=1, help="Maximum transforms composed in each robust pass")
    parser.add_argument(
        "--forensic-mode", choices=("laplacian", "fft", "laplacian_fft"), default="laplacian"
    )
    parser.add_argument(
        "--modality-dropout", type=float, default=0.0, help="Chance of zeroing CLIP or forensic features per head batch"
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if (requested == "auto" and torch.cuda.is_available()) or requested == "cuda" else "cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    rows = load_labeled_paths(args.data_dir)
    train_rows, val_rows, test_rows = stratified_train_val_test_split(
        rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
    )
    if not 0.0 <= args.modality_dropout < 1.0:
        raise ValueError("--modality-dropout must be in [0, 1)")
    forensic_dim = 2560 if args.forensic_mode == "laplacian_fft" else 1280
    config = ModelConfig(forensic_mode=args.forensic_mode, forensic_dim=forensic_dim)

    if args.cache and args.cache.exists():
        cached = torch.load(args.cache, map_location="cpu", weights_only=True)
        train_x, train_y = cached["train_features"], cached["train_labels"]
        val_x, val_y = cached["val_features"], cached["val_labels"]
        test_x, test_y = cached.get("test_features"), cached.get("test_labels")
        if train_x.shape[1] != config.clip_dim + config.forensic_dim:
            raise ValueError(
                f"Feature cache has width {train_x.shape[1]}, but {args.forensic_mode!r} expects "
                f"{config.clip_dim + config.forensic_dim}; use a mode-specific cache path"
            )
        print(f"Loaded feature cache: {args.cache}")
    else:
        encoders = FrozenEncoders(config, device)
        train_x, train_y, _ = extract_features(
            train_rows,
            encoders,
            args.feature_batch_size,
            max(1, args.augmentation_repeats),
            robust=True,
            augmentation_depth=args.augmentation_depth,
        )
        val_x, val_y, _ = extract_features(val_rows, encoders, args.feature_batch_size)
        test_x = test_y = None
        if args.report_test_metrics:
            test_x, test_y, _ = extract_features(test_rows, encoders, args.feature_batch_size)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "train_features": train_x,
                    "train_labels": train_y,
                    "val_features": val_x,
                    "val_labels": val_y,
                    **(
                        {"test_features": test_x, "test_labels": test_y}
                        if test_x is not None and test_y is not None
                        else {}
                    ),
                },
                args.cache,
            )
            print(f"Saved feature cache: {args.cache}")
        del encoders

    head = FusionHead(config).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.head_batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, stale = float("inf"), 0
    val_x_device, val_y_device = val_x.to(device), val_y.to(device)
    for epoch in range(1, args.epochs + 1):
        head.train()
        for features, labels in loader:
            if args.modality_dropout > 0:
                features = features.clone()
                drop_clip = torch.rand(len(features)) < (args.modality_dropout / 2)
                drop_forensic = (~drop_clip) & (torch.rand(len(features)) < (args.modality_dropout / 2))
                features[drop_clip, : config.clip_dim] = 0
                features[drop_forensic, config.clip_dim :] = 0
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(val_x_device)
            val_loss = float(criterion(val_logits, val_y_device))
        print(f"epoch={epoch:03d} val_loss={val_loss:.5f}")
        if val_loss < best_loss - 1e-5:
            best_loss, stale = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    assert best_state is not None
    head.load_state_dict(best_state)
    head.to("cpu").eval()
    with torch.no_grad():
        logits = head(val_x)
    temperature = fit_temperature(logits, val_y)
    metrics = classification_metrics(val_y, torch.sigmoid(logits / temperature))
    test_metrics = None
    if args.report_test_metrics:
        if test_x is None or test_y is None:
            encoders = FrozenEncoders(config, device)
            test_x, test_y, _ = extract_features(test_rows, encoders, args.feature_batch_size)
            del encoders
        with torch.no_grad():
            test_metrics = classification_metrics(test_y, torch.sigmoid(head(test_x) / temperature))
    metadata = {
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "test_images": len(test_rows),
        "train_feature_rows": len(train_y),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "forensic_mode": args.forensic_mode,
        "augmentation_repeats": args.augmentation_repeats,
        "augmentation_depth": args.augmentation_depth,
        "modality_dropout": args.modality_dropout,
        "validation_metrics": metrics,
        **({"test_metrics": test_metrics} if test_metrics is not None else {}),
    }
    save_checkpoint(args.output, head, config, temperature, metadata)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "temperature": temperature,
                "validation": metrics,
                **({"test": test_metrics} if test_metrics is not None else {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
