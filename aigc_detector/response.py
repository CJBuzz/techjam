from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from .data import BALANCED_TRANSFORM_GROUPS, DeterministicTransform, ROBUST_SELECTION_CONDITIONS

PROBE_CONDITIONS = ("clean", "jpeg_q90", "blur_s0.5", "resize_x0.5")
RESPONSE_FEATURE_DIM = 20


def perturbation_views(image: Image.Image, seed: int, identity: str) -> list[Image.Image]:
    """Return x and three deterministic mild redistribution probes."""
    return [
        image.copy() if condition == "clean" else DeterministicTransform(
            condition, seed, identity, index
        )(image.copy())
        for index, condition in enumerate(PROBE_CONDITIONS)
    ]


def response_features(
    encoder_features: torch.Tensor,
    calibrated_logits: torch.Tensor,
    clip_dim: int = 512,
) -> torch.Tensor:
    """Build compact logit-response and modality-drift features for four views."""
    if encoder_features.ndim != 3 or encoder_features.shape[1] != 4:
        raise ValueError("Expected encoder features shaped [images, 4 views, features]")
    if calibrated_logits.shape != encoder_features.shape[:2]:
        raise ValueError("Calibrated logits must align with image/view dimensions")
    if encoder_features.shape[2] < clip_dim + 2560:
        raise ValueError("Response features require CLIP + Laplacian + FFT encoder blocks")
    logit_statistics = torch.cat(
        (
            calibrated_logits,
            calibrated_logits.mean(1, keepdim=True),
            calibrated_logits.std(1, unbiased=False, keepdim=True),
            calibrated_logits.min(1, keepdim=True).values,
            calibrated_logits.max(1, keepdim=True).values,
            calibrated_logits[:, 1:] - calibrated_logits[:, :1],
        ),
        dim=1,
    )
    modality_slices = (
        slice(0, clip_dim),
        slice(clip_dim, clip_dim + 1280),
        slice(clip_dim + 1280, clip_dim + 2560),
    )
    drifts = []
    for modality in modality_slices:
        base = encoder_features[:, :1, modality]
        transformed = encoder_features[:, 1:, modality]
        drifts.append(1 - F.cosine_similarity(base.expand_as(transformed), transformed, dim=2))
    result = torch.cat((logit_statistics, *drifts), dim=1)
    if result.shape[1] != RESPONSE_FEATURE_DIM:
        raise AssertionError(f"Unexpected response feature width: {result.shape[1]}")
    return result


class ResponseHead(nn.Module):
    """Tiny residual correction initialized to preserve the base calibrated logit."""

    def __init__(self, input_dim: int = RESPONSE_FEATURE_DIM, hidden_dim: int = 32) -> None:
        super().__init__()
        self.correction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0] + self.correction(features).squeeze(1)


def restore_best_response_state(head: ResponseHead, state: dict[str, torch.Tensor]) -> None:
    """Strictly restore the selected state, then switch the model—not the load result—to eval."""
    result = head.load_state_dict(state)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Best E3 state is incompatible: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    head.eval()


def extract_response_features(
    rows: list[tuple[Path, int]],
    conditions: list[str],
    encoders: nn.Module,
    base_head: nn.Module,
    base_temperature: float,
    clip_dim: int,
    feature_dim: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Extract each official input once, then encode all four response views together."""
    if len(rows) != len(conditions):
        raise ValueError("Each row requires one input condition")
    output, labels, paths = [], [], []
    for start in range(0, len(rows), batch_size):
        images = []
        batch_rows = rows[start : start + batch_size]
        batch_conditions = conditions[start : start + batch_size]
        for (path, label), condition in zip(batch_rows, batch_conditions, strict=True):
            with Image.open(path) as source:
                conditioned = DeterministicTransform(condition, seed, str(path), 0)(source.convert("RGB"))
            images.extend(perturbation_views(conditioned, seed, f"{path}:{condition}"))
            labels.append(label)
            paths.append(str(path))
        encoded = encoders(images)[..., :feature_dim].view(len(batch_rows), 4, feature_dim)
        with torch.no_grad():
            logits = base_head(encoded.flatten(0, 1).to(device)).view(len(batch_rows), 4).cpu()
        output.append(response_features(encoded, logits / base_temperature, clip_dim))
    return torch.cat(output), torch.tensor(labels, dtype=torch.float32), paths


def load_response_checkpoint(path: Path, device: torch.device) -> tuple[ResponseHead, float, float, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    head = ResponseHead(payload["input_dim"], payload["hidden_dim"]).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, float(payload["temperature"]), float(payload["threshold"]), payload["metadata"]


def _fingerprint(rows: list[tuple[Path, int]], root: Path) -> str:
    digest = hashlib.sha256()
    for path, label in sorted(rows, key=lambda row: str(row[0])):
        stat = path.stat()
        digest.update(
            f"{path.resolve().relative_to(root.resolve())}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def extract_command(args: argparse.Namespace) -> None:
    from dataclasses import asdict

    from .data import load_labeled_paths, stratified_train_val_test_split
    from .model import FrozenEncoders, load_checkpoint
    from .train import choose_device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    rows = load_labeled_paths(args.data_dir)
    train_rows, val_rows, _ = stratified_train_val_test_split(
        rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
    )
    base_head, config, base_temperature, base_metadata = load_checkpoint(args.base_checkpoint, device)
    if config.forensic_mode != "laplacian_fft" or config.forensic_dim != 2560:
        raise ValueError("E3 requires a Laplacian+FFT base checkpoint")
    manifest = {
        "schema_version": 1,
        "dataset_fingerprint": _fingerprint(rows, args.data_dir),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_config": asdict(config),
        "base_temperature": base_temperature,
        "base_threshold": float(base_metadata.get("threshold", 0.5)),
        "seed": args.seed,
        "augmentation_repeats": args.augmentation_repeats,
        "validation_conditions": ["clean", *ROBUST_SELECTION_CONDITIONS],
        "test_rows_extracted": False,
    }
    if args.output.exists():
        cached = torch.load(args.output, map_location="cpu", weights_only=True)
        if cached.get("manifest") != manifest:
            raise ValueError("Existing E3 feature cache does not match requested settings")
        print(f"E3 feature cache already exists: {args.output}")
        return
    encoders = FrozenEncoders(config, device)
    train_features, train_labels, train_groups = [], [], []
    for repeat in range(args.augmentation_repeats):
        conditions = [
            "clean" if repeat == 0 else BALANCED_TRANSFORM_GROUPS[
                (index * (args.augmentation_repeats - 1) + repeat - 1) % len(BALANCED_TRANSFORM_GROUPS)
            ]
            for index in range(len(train_rows))
        ]
        features, labels, _ = extract_response_features(
            train_rows, conditions, encoders, base_head, base_temperature,
            config.clip_dim, config.clip_dim + config.forensic_dim, args.batch_size, args.seed, device,
        )
        train_features.append(features)
        train_labels.append(labels)
        train_groups.extend(conditions)
    val_features, val_labels, val_groups = [], [], []
    for condition in ("clean", *ROBUST_SELECTION_CONDITIONS):
        features, labels, _ = extract_response_features(
            val_rows, [condition] * len(val_rows), encoders, base_head, base_temperature,
            config.clip_dim, config.clip_dim + config.forensic_dim, args.batch_size, args.seed, device,
        )
        val_features.append(features)
        val_labels.append(labels)
        val_groups.extend([condition] * len(val_rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "train_features": torch.cat(train_features),
        "train_labels": torch.cat(train_labels),
        "train_groups": train_groups,
        "val_features": torch.cat(val_features),
        "val_labels": torch.cat(val_labels),
        "val_groups": val_groups,
        "manifest": manifest,
    }, args.output)
    print(f"Saved E3 response-feature cache: {args.output}")


def train_command(args: argparse.Namespace) -> None:
    from .metrics import classification_metrics, fit_temperature, select_threshold

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    train_x, train_y = cache["train_features"], cache["train_labels"]
    val_x, val_y, val_groups = cache["val_features"], cache["val_labels"], cache["val_groups"]
    head = ResponseHead(train_x.shape[1], args.hidden_dim)
    if sum(parameter.numel() for parameter in head.parameters()) >= 100_000:
        raise ValueError("Response head must remain below 100k parameters")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    groups = sorted(set(val_groups))
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            logits = head(val_x)
            losses = []
            for group in groups:
                mask = torch.tensor([value == group for value in val_groups], dtype=torch.bool)
                losses.append(criterion(logits[mask], val_y[mask]))
            selection_loss = 0.5 * (torch.stack(losses).mean() + torch.stack(losses).max())
        if float(selection_loss) < best_loss - 1e-5:
            best_loss, stale = float(selection_loss), 0
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
        print(f"epoch={epoch:03d} robust_selection={float(selection_loss):.5f}")
    if best_state is None:
        raise RuntimeError("E3 training produced no checkpoint")
    restore_best_response_state(head, best_state)
    with torch.no_grad():
        logits = head(val_x)
    temperature = fit_temperature(logits, val_y)
    probabilities = torch.sigmoid(logits / temperature)
    threshold = select_threshold(val_y, probabilities, "balanced")
    validation = {
        group: classification_metrics(
            val_y[torch.tensor([value == group for value in val_groups])],
            probabilities[torch.tensor([value == group for value in val_groups])],
            threshold,
        )
        for group in groups
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(),
        "input_dim": train_x.shape[1],
        "hidden_dim": args.hidden_dim,
        "temperature": temperature,
        "threshold": threshold,
        "metadata": {
            "base_checkpoint": cache["manifest"]["base_checkpoint"],
            "cache_manifest": cache["manifest"],
            "train_rows": len(train_y),
            "validation_rows": len(val_y),
            "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
            "validation_metrics": validation,
            "test_rows_used": False,
        },
    }, args.output)
    print(json.dumps({"checkpoint": str(args.output), "temperature": temperature, "threshold": threshold}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 perturbation-response experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--data-dir", type=Path, required=True)
    extract.add_argument("--base-checkpoint", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--augmentation-repeats", type=int, default=2)
    extract.add_argument("--validation-fraction", type=float, default=0.15)
    extract.add_argument("--test-fraction", type=float, default=0.15)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    extract.set_defaults(handler=extract_command)
    train = subparsers.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--hidden-dim", type=int, default=32)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(handler=train_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
