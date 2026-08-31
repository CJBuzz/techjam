"""Train and validation-select the small fusion head over frozen feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    ROBUST_SELECTION_CONDITIONS,
    load_labeled_paths,
    load_split_manifest,
    stratified_train_val_test_split,
)
from .features import extract_balanced_features, extract_condition_features, extract_features
from .metrics import classification_metrics, fit_temperature, select_threshold
from .model import FrozenEncoders, FusionHead, ModelConfig, load_checkpoint, save_checkpoint


def merge_balanced_feature_sets(
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    local_groups: list[str],
    local_originals: int,
    diverse_x: torch.Tensor,
    diverse_y: torch.Tensor,
    diverse_groups: list[str],
    diverse_originals: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, int]:
    """Merge paired local and streamed features while preserving repeat-major pairing."""
    local_repeats = len(local_y) // local_originals
    diverse_repeats = len(diverse_y) // diverse_originals
    # Both sources must contribute the same complete view count per original.
    if local_repeats != diverse_repeats:
        raise ValueError(
            f"Local cache has {local_repeats} paired views, diverse cache has {diverse_repeats}"
        )
    merged_x = torch.cat(
        (local_x.view(local_repeats, local_originals, -1),
         diverse_x.view(local_repeats, diverse_originals, -1)), dim=1,
    ).flatten(0, 1)
    merged_y = torch.cat(
        (local_y.view(local_repeats, local_originals),
         diverse_y.view(local_repeats, diverse_originals)), dim=1,
    ).flatten()
    merged_groups = [
        group
        for repeat in range(local_repeats)
        for group in (
            local_groups[repeat * local_originals : (repeat + 1) * local_originals]
            + diverse_groups[repeat * diverse_originals : (repeat + 1) * diverse_originals]
        )
    ]
    total_originals = local_originals + diverse_originals
    # Rebuild identities after concatenation so unrelated sources never pair.
    original_indices = torch.arange(total_originals).repeat(local_repeats)
    return merged_x, merged_y, merged_groups, original_indices, total_originals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen two-stream AIGC detector")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing real/ and ai/ folders")
    parser.add_argument("--split-manifest", type=Path, default=None, help="Persisted duplicate-aware split manifest")
    parser.add_argument("--output", type=Path, default=Path("artifacts/trained_detector.pt"))
    parser.add_argument("--cache", type=Path, default=None, help="Optional feature cache (.pt)")
    parser.add_argument(
        "--diverse-cache", type=Path, default=None,
        help="Optional streamed CommunityForensics feature-cache directory added to local training only",
    )
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
        "--augmentation-policy", choices=("random", "balanced"), default="random",
        help="Balanced creates deterministic paired views with explicit transform groups",
    )
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--worst-group-weight", type=float, default=0.0)
    parser.add_argument(
        "--robust-validation", action=argparse.BooleanOptionalAction, default=True,
        help="Select checkpoints on clean plus deterministic worst-severity validation views",
    )
    parser.add_argument(
        "--robust-validation-weight", type=float, default=0.7,
        help="Weight assigned to robust mean/worst validation loss during checkpoint selection",
    )
    parser.add_argument("--threshold-objective", choices=("balanced", "f1"), default="balanced")
    parser.add_argument(
        "--forensic-mode", choices=("laplacian", "fft", "laplacian_fft"), default="laplacian"
    )
    parser.add_argument(
        "--modality-dropout", type=float, default=0.0, help="Chance of zeroing CLIP or forensic features per head batch"
    )
    parser.add_argument(
        "--fft-dropout",
        type=float,
        default=0.0,
        help="For laplacian_fft only: chance of masking the FFT feature block per training sample",
    )
    parser.add_argument(
        "--initialize-from-laplacian",
        type=Path,
        default=None,
        help="Initialize a laplacian_fft head from a compatible Laplacian checkpoint with zero FFT weights",
    )
    parser.add_argument(
        "--initialize-from-checkpoint",
        type=Path,
        default=None,
        help="Initialize the head from a checkpoint with an exactly matching model configuration",
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu")
    return torch.device(requested)


def _dataset_fingerprint(rows: list[tuple[Path, int]], root: Path) -> str:
    """Fast cache identity based on paths, labels, sizes, and modification times."""
    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for path, label in sorted(rows, key=lambda row: str(row[0])):
        stat = path.stat()
        relative = path.resolve().relative_to(resolved_root)
        digest.update(f"{relative}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def build_cache_manifest(args: argparse.Namespace, config: ModelConfig, rows: list[tuple[Path, int]]) -> dict:
    split_manifest_hash = None
    if args.split_manifest:
        # Bind caches to the exact persisted split, not only the dataset directory.
        split_manifest_hash = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    return {
        "schema_version": 2,
        "dataset_fingerprint": _dataset_fingerprint(rows, args.data_dir),
        "model_config": asdict(config),
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "split_manifest_sha256": split_manifest_hash,
        "augmentation_policy": args.augmentation_policy,
        "augmentation_repeats": max(2, args.augmentation_repeats)
        if args.augmentation_policy == "balanced" else max(1, args.augmentation_repeats),
        "augmentation_depth": args.augmentation_depth,
        "robust_validation": args.robust_validation,
        "robust_validation_conditions": list(ROBUST_SELECTION_CONDITIONS) if args.robust_validation else [],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    rows = load_labeled_paths(args.data_dir)
    calibration_rows: list[tuple[Path, int]] = []
    if args.split_manifest:
        # Four-way manifests keep checkpoint selection and calibration independent.
        manifest_splits = load_split_manifest(args.data_dir, args.split_manifest)
        train_rows, val_rows, calibration_rows, test_rows = (
            manifest_splits["train"],
            manifest_splits["model_selection"],
            manifest_splits["calibration"],
            manifest_splits["test"],
        )
    else:
        train_rows, val_rows, test_rows = stratified_train_val_test_split(
            rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
        )
    if not 0.0 <= args.modality_dropout < 1.0:
        raise ValueError("--modality-dropout must be in [0, 1)")
    if not 0.0 <= args.fft_dropout < 1.0:
        raise ValueError("--fft-dropout must be in [0, 1)")
    if args.fft_dropout and args.forensic_mode != "laplacian_fft":
        raise ValueError("--fft-dropout requires --forensic-mode laplacian_fft")
    forensic_dim = 2560 if args.forensic_mode == "laplacian_fft" else 1280
    config = ModelConfig(forensic_mode=args.forensic_mode, forensic_dim=forensic_dim)
    if args.consistency_weight < 0 or args.worst_group_weight < 0:
        raise ValueError("Consistency and worst-group weights must be non-negative")
    if not 0.0 <= args.robust_validation_weight <= 1.0:
        raise ValueError("--robust-validation-weight must be in [0, 1]")
    if (args.consistency_weight or args.worst_group_weight) and args.augmentation_policy != "balanced":
        raise ValueError("Paired consistency and worst-group loss require --augmentation-policy balanced")

    expected_manifest = build_cache_manifest(args, config, rows)
    if args.cache and args.cache.exists():
        # Refuse reuse if data, split, encoder, or augmentation settings changed.
        cached = torch.load(args.cache, map_location="cpu", weights_only=True)
        if cached.get("manifest") != expected_manifest:
            raise ValueError(
                "Feature cache does not match the dataset, split, encoder, or augmentation settings; "
                "use a new cache path or rebuild it"
            )
        train_x, train_y = cached["train_features"], cached["train_labels"]
        val_x, val_y = cached["val_features"], cached["val_labels"]
        calibration_x = cached.get("calibration_features")
        calibration_y = cached.get("calibration_labels")
        test_x, test_y = cached.get("test_features"), cached.get("test_labels")
        robust_val_x, robust_val_y = cached.get("robust_val_features"), cached.get("robust_val_labels")
        robust_val_conditions = cached.get("robust_val_conditions")
        train_groups = cached.get("train_groups")
        train_original_indices = cached.get("train_original_indices")
        cache_policy = cached.get("augmentation_policy", "random")
        if args.augmentation_policy != cache_policy:
            raise ValueError(f"Feature cache policy is {cache_policy!r}, requested {args.augmentation_policy!r}")
        if train_x.shape[1] != config.clip_dim + config.forensic_dim:
            raise ValueError(
                f"Feature cache has width {train_x.shape[1]}, but {args.forensic_mode!r} expects "
                f"{config.clip_dim + config.forensic_dim}; use a mode-specific cache path"
            )
        print(f"Loaded feature cache: {args.cache}")
    else:
        # Encoder extraction is expensive; head experiments should reuse this cache.
        encoders = FrozenEncoders(config, device)
        if args.augmentation_policy == "balanced":
            train_x, train_y, _, train_groups, train_original_indices = extract_balanced_features(
                train_rows, encoders, args.feature_batch_size, max(2, args.augmentation_repeats), args.seed
            )
        else:
            train_x, train_y, _ = extract_features(
                train_rows, encoders, args.feature_batch_size, max(1, args.augmentation_repeats), robust=True,
                augmentation_depth=args.augmentation_depth,
            )
            train_groups = train_original_indices = None
        val_x, val_y, _ = extract_features(val_rows, encoders, args.feature_batch_size)
        robust_val_x = robust_val_y = robust_val_conditions = None
        if args.robust_validation:
            robust_val_x, robust_val_y, _, robust_val_conditions = extract_condition_features(
                val_rows, encoders, args.feature_batch_size, ROBUST_SELECTION_CONDITIONS, args.seed
            )
        calibration_x = calibration_y = None
        if calibration_rows:
            calibration_x, calibration_y, _ = extract_features(
                calibration_rows, encoders, args.feature_batch_size
            )
        test_x = test_y = None
        if args.report_test_metrics:
            test_x, test_y, _ = extract_features(test_rows, encoders, args.feature_batch_size)
        if args.cache:
            # Test tensors are omitted unless the caller explicitly opens that split.
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "train_features": train_x,
                    "train_labels": train_y,
                    "val_features": val_x,
                    "val_labels": val_y,
                    "manifest": expected_manifest,
                    "augmentation_policy": args.augmentation_policy,
                    **(
                        {"calibration_features": calibration_x, "calibration_labels": calibration_y}
                        if calibration_x is not None and calibration_y is not None
                        else {}
                    ),
                    **(
                        {
                            "robust_val_features": robust_val_x,
                            "robust_val_labels": robust_val_y,
                            "robust_val_conditions": robust_val_conditions,
                        }
                        if robust_val_x is not None and robust_val_y is not None else {}
                    ),
                    **(
                        {"train_groups": train_groups, "train_original_indices": train_original_indices}
                        if train_groups is not None and train_original_indices is not None else {}
                    ),
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

    training_original_count = len(train_rows)
    diverse_manifest = None
    if args.diverse_cache:
        if args.augmentation_policy != "balanced":
            raise ValueError("--diverse-cache requires --augmentation-policy balanced")
        if train_groups is None or train_original_indices is None:
            raise ValueError("Local balanced cache is missing group/pair metadata")
        from .tooling.streaming_cache import load_stream_feature_cache

        diverse_x, diverse_y, diverse_groups, _, diverse_originals, diverse_manifest = load_stream_feature_cache(
            args.diverse_cache, asdict(config)
        )
        # Extend the original-ID namespace while preserving complete paired groups.
        train_x, train_y, train_groups, train_original_indices, training_original_count = (
            merge_balanced_feature_sets(
                train_x, train_y, train_groups, len(train_rows),
                diverse_x, diverse_y, diverse_groups, diverse_originals,
            )
        )
        print(f"Added {diverse_originals} streamed originals from: {args.diverse_cache}")

    head = FusionHead(config).to(device)
    if args.initialize_from_checkpoint and args.initialize_from_laplacian:
        raise ValueError("Choose only one checkpoint initialization mode")
    if args.initialize_from_checkpoint:
        source_head, source_config, _, _ = load_checkpoint(
            args.initialize_from_checkpoint, torch.device("cpu")
        )
        if asdict(source_config) != asdict(config):
            raise ValueError("Initialization checkpoint model configuration does not match E1")
        head.load_state_dict(source_head.state_dict())
        print(f"Initialized head from: {args.initialize_from_checkpoint}")
    if args.initialize_from_laplacian:
        if args.forensic_mode != "laplacian_fft":
            raise ValueError("--initialize-from-laplacian requires --forensic-mode laplacian_fft")
        source_head, source_config, _, _ = load_checkpoint(args.initialize_from_laplacian, torch.device("cpu"))
        if source_config.forensic_mode != "laplacian" or source_config.clip_dim != config.clip_dim:
            raise ValueError("Initialization checkpoint must use the compatible Laplacian encoder")
        source_state = source_head.state_dict()
        target_state = head.state_dict()
        # Copy CLIP+Laplacian weights and introduce FFT as a zero-valued residual.
        for key, source_value in source_state.items():
            if key == "network.0.weight":
                target_state[key].zero_()
                target_state[key][:, : source_value.shape[1]].copy_(source_value)
            elif target_state[key].shape == source_value.shape:
                target_state[key].copy_(source_value)
        head.load_state_dict(target_state)
        print(f"Initialized CLIP+Laplacian head weights from: {args.initialize_from_laplacian}")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    if args.augmentation_policy == "balanced":
        if train_groups is None or train_original_indices is None:
            raise ValueError("Balanced feature cache is missing group/pair metadata")
        repeats = len(train_y) // training_original_count
        # Sample originals first; each minibatch then gathers every paired view.
        if repeats * training_original_count != len(train_y):
            raise ValueError("Balanced feature rows must contain complete repeats of every original")
        original_loader = DataLoader(torch.arange(training_original_count), batch_size=args.head_batch_size, shuffle=True)
        group_names = sorted(set(train_groups))
    else:
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.head_batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, stale = float("inf"), 0
    val_x_device, val_y_device = val_x.to(device), val_y.to(device)
    robust_val_x_device = robust_val_x.to(device) if robust_val_x is not None else None
    robust_val_y_device = robust_val_y.to(device) if robust_val_y is not None else None
    for epoch in range(1, args.epochs + 1):
        head.train()
        batches = original_loader if args.augmentation_policy == "balanced" else loader
        for batch in batches:
            if args.augmentation_policy == "balanced":
                original_batch = batch
                feature_indices = torch.cat([
                    original_batch + repeat * training_original_count for repeat in range(repeats)
                ])
                features, labels = train_x[feature_indices], train_y[feature_indices]
                batch_groups = [train_groups[index] for index in feature_indices.tolist()]
            else:
                features, labels = batch
            if args.modality_dropout > 0:
                # Mask whole modalities so the head learns meaningful fallback cues.
                features = features.clone()
                drop_clip = torch.rand(len(features)) < (args.modality_dropout / 2)
                drop_forensic = (~drop_clip) & (torch.rand(len(features)) < (args.modality_dropout / 2))
                features[drop_clip, : config.clip_dim] = 0
                features[drop_forensic, config.clip_dim :] = 0
            if args.fft_dropout > 0:
                # FFT has its own dropout because it was added as a residual block.
                fft_start = config.clip_dim + 1280
                drop_fft = torch.rand(len(features)) < args.fft_dropout
                features[drop_fft, fft_start:] = 0
            optimizer.zero_grad(set_to_none=True)
            batch_logits = head(features.to(device))
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                batch_logits, labels.to(device), reduction="none"
            )
            loss = losses.mean()
            if args.consistency_weight > 0:
                # Paired corruptions should agree in logit space, not raw feature space.
                paired_logits = batch_logits.view(repeats, -1)
                loss = loss + args.consistency_weight * ((paired_logits - paired_logits.mean(0)) ** 2).mean()
            if args.worst_group_weight > 0:
                # Optional minimax pressure targets the hardest group in this batch.
                group_losses = []
                for group in group_names:
                    mask = torch.tensor([value == group for value in batch_groups], device=device)
                    if mask.any():
                        group_losses.append(losses[mask].mean())
                loss = loss + args.worst_group_weight * torch.stack(group_losses).max()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(val_x_device)
            clean_val_loss = criterion(val_logits, val_y_device)
            robust_mean_loss = robust_worst_loss = clean_val_loss
            if robust_val_x_device is not None and robust_val_y_device is not None:
                # Balance average robustness against the single hardest condition.
                robust_logits = head(robust_val_x_device)
                condition_size = len(val_rows)
                condition_losses = torch.stack([
                    criterion(
                        robust_logits[start : start + condition_size],
                        robust_val_y_device[start : start + condition_size],
                    )
                    for start in range(0, len(robust_logits), condition_size)
                ])
                robust_mean_loss = condition_losses.mean()
                robust_worst_loss = condition_losses.max()
            robust_selection_loss = 0.5 * (robust_mean_loss + robust_worst_loss)
            selection_loss = (
                (1 - args.robust_validation_weight) * clean_val_loss
                + args.robust_validation_weight * robust_selection_loss
            )
        print(
            f"epoch={epoch:03d} clean={float(clean_val_loss):.5f} "
            f"robust_mean={float(robust_mean_loss):.5f} robust_worst={float(robust_worst_loss):.5f} "
            f"selection={float(selection_loss):.5f}"
        )
        if float(selection_loss) < best_loss - 1e-5:
            # CPU snapshots avoid retaining a live graph or unnecessary device memory.
            best_loss, stale = float(selection_loss), 0
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
        clean_logits = head(val_x)
        if calibration_x is not None and calibration_y is not None:
            # A persisted four-way manifest keeps calibration independent from
            # checkpoint selection. The dedicated calibration CLI can add exact
            # transformed views later without touching model-selection data.
            calibration_logits = head(calibration_x)
            calibration_labels = calibration_y
        else:
            # Legacy three-way datasets explicitly fall back to validation calibration.
            calibration_logits = clean_logits
            calibration_labels = val_y
        if calibration_x is None and robust_val_x is not None and robust_val_y is not None:
            calibration_logits = torch.cat((clean_logits, head(robust_val_x)))
            calibration_labels = torch.cat((val_y, robust_val_y))
    temperature = fit_temperature(calibration_logits, calibration_labels)
    # Threshold fitting uses calibration labels and never reserved-test labels.
    calibration_probabilities = torch.sigmoid(calibration_logits / temperature)
    threshold = select_threshold(calibration_labels, calibration_probabilities, args.threshold_objective)
    metrics = classification_metrics(val_y, torch.sigmoid(clean_logits / temperature), threshold)
    robust_validation_metrics = None
    if robust_val_x is not None and robust_val_y is not None:
        with torch.no_grad():
            robust_probabilities = torch.sigmoid(head(robust_val_x) / temperature)
        robust_validation_metrics = {}
        for condition in ROBUST_SELECTION_CONDITIONS:
            mask = torch.tensor([value == condition for value in robust_val_conditions], dtype=torch.bool)
            robust_validation_metrics[condition] = classification_metrics(
                robust_val_y[mask], robust_probabilities[mask], threshold
            )
    test_metrics = None
    if args.report_test_metrics:
        if test_x is None or test_y is None:
            encoders = FrozenEncoders(config, device)
            test_x, test_y, _ = extract_features(test_rows, encoders, args.feature_batch_size)
            del encoders
        with torch.no_grad():
            test_metrics = classification_metrics(test_y, torch.sigmoid(head(test_x) / temperature), threshold)
    metadata = {
        # Store enough provenance to audit and reproduce the deployable head.
        "train_images": len(train_rows),
        "diverse_train_images": training_original_count - len(train_rows),
        "diverse_cache": str(args.diverse_cache) if args.diverse_cache else None,
        "diverse_cache_manifest": diverse_manifest,
        "validation_images": len(val_rows),
        "test_images": len(test_rows),
        "train_feature_rows": len(train_y),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "calibration_images": len(calibration_y) if calibration_y is not None else len(val_y),
        "calibration_split": "calibration" if calibration_y is not None else "validation",
        "forensic_mode": args.forensic_mode,
        "augmentation_repeats": args.augmentation_repeats,
        "augmentation_depth": args.augmentation_depth,
        "augmentation_policy": args.augmentation_policy,
        "consistency_weight": args.consistency_weight,
        "worst_group_weight": args.worst_group_weight,
        "robust_validation": args.robust_validation,
        "robust_validation_weight": args.robust_validation_weight,
        "robust_validation_metrics": robust_validation_metrics,
        "threshold": threshold,
        "threshold_objective": args.threshold_objective,
        "modality_dropout": args.modality_dropout,
        "fft_dropout": args.fft_dropout,
        "initialized_from_checkpoint": str(args.initialize_from_checkpoint)
        if args.initialize_from_checkpoint else None,
        "initialized_from_laplacian": str(args.initialize_from_laplacian) if args.initialize_from_laplacian else None,
        "validation_metrics": metrics,
        **({"test_metrics": test_metrics} if test_metrics is not None else {}),
    }
    save_checkpoint(args.output, head, config, temperature, metadata)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "temperature": temperature,
                "threshold": threshold,
                "validation": metrics,
                "robust_validation": robust_validation_metrics,
                **({"test": test_metrics} if test_metrics is not None else {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
