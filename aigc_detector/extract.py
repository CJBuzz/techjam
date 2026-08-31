"""Feature-cache stage for the reproducible large-corpus training pipeline.

The command reads train, model-selection, and calibration rows from a persisted
split manifest. Reserved-test images are intentionally never decoded. It writes
both CLIP+Laplacian+FFT features and a CLIP+Laplacian initializer slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import ROBUST_SELECTION_CONDITIONS, load_labeled_paths, load_split_manifest
from .features import extract_balanced_features, extract_condition_features, extract_features
from .model import FrozenEncoders, ModelConfig
from .train import build_cache_manifest, choose_device


def _save_cache(path: Path, payload: dict) -> None:
    """Persist features and explicit provenance for compatibility checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_cache(
    path: Path,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    calibration_x: torch.Tensor,
    calibration_y: torch.Tensor,
    robust_val_x: torch.Tensor,
    robust_val_y: torch.Tensor,
    robust_val_conditions: list[str],
    groups: list[str],
    original_indices: torch.Tensor,
    manifest: dict,
    split_manifest: Path,
    forensic_mode: str,
) -> None:
    """Write the scale-cache schema used by training and Kaggle handoffs."""
    # Labels and pair metadata travel with features so training can audit alignment.
    _save_cache(path, {
        "train_features": train_x,
        "train_labels": train_y,
        "val_features": val_x,
        "val_labels": val_y,
        "calibration_features": calibration_x,
        "calibration_labels": calibration_y,
        "robust_val_features": robust_val_x,
        "robust_val_labels": robust_val_y,
        "robust_val_conditions": robust_val_conditions,
        "train_groups": groups,
        "train_original_indices": original_indices,
        "augmentation_policy": "balanced",
        "manifest": manifest,
        "split_manifest": str(split_manifest),
        "forensic_mode": forensic_mode,
        # This explicit audit flag prevents consumers from assuming test availability.
        "test_features_extracted": False,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract submission-model features without reading the reserved test split"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--laplacian-output", type=Path, required=True)
    parser.add_argument("--augmentation-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_split_manifest(args.data_dir, args.split_manifest)
    # All rows are used only for the fingerprint; reserved-test pixels stay unread.
    all_rows = load_labeled_paths(args.data_dir)
    device = choose_device(args.device)
    config = ModelConfig(forensic_mode="laplacian_fft", forensic_dim=2560)
    encoders = FrozenEncoders(config, device)

    # Shared original indices keep all transformed views together for paired
    # consistency training and make split leakage structurally impossible.
    train_x, train_y, _, groups, original_indices = extract_balanced_features(
        splits["train"], encoders, args.batch_size, args.augmentation_repeats, args.seed
    )
    val_x, val_y, _ = extract_features(splits["model_selection"], encoders, args.batch_size)
    calibration_x, calibration_y, _ = extract_features(splits["calibration"], encoders, args.batch_size)
    # Robust selection diagnostics never reuse the independent calibration rows.
    robust_val_x, robust_val_y, _, robust_val_conditions = extract_condition_features(
        splits["model_selection"], encoders, args.batch_size, ROBUST_SELECTION_CONDITIONS, args.seed
    )
    cache_args = argparse.Namespace(
        data_dir=args.data_dir,
        split_manifest=args.split_manifest,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=args.seed,
        augmentation_policy="balanced",
        augmentation_repeats=args.augmentation_repeats,
        augmentation_depth=1,
        robust_validation=True,
    )

    common = {
        "train_labels": train_y,
        "val_labels": val_y,
        "calibration_labels": calibration_y,
        "robust_val_labels": robust_val_y,
        "robust_val_conditions": robust_val_conditions,
        "train_groups": groups,
        "train_original_indices": original_indices,
        "augmentation_policy": "balanced",
        "split_manifest": str(args.split_manifest),
        "test_features_extracted": False,
    }
    # Both caches share identical row ordering and provenance metadata.
    _save_cache(
        args.combined_output,
        {
            **common,
            "train_features": train_x,
            "val_features": val_x,
            "calibration_features": calibration_x,
            "robust_val_features": robust_val_x,
            "manifest": build_cache_manifest(cache_args, config, all_rows),
            "forensic_mode": "laplacian_fft",
        },
    )
    laplacian_width = config.clip_dim + 1280
    # Laplacian features are an exact prefix, avoiding a second encoder pass.
    laplacian_config = ModelConfig(forensic_mode="laplacian", forensic_dim=1280)
    _save_cache(
        args.laplacian_output,
        {
            **common,
            "train_features": train_x[:, :laplacian_width],
            "val_features": val_x[:, :laplacian_width],
            "calibration_features": calibration_x[:, :laplacian_width],
            "robust_val_features": robust_val_x[:, :laplacian_width],
            "manifest": build_cache_manifest(cache_args, laplacian_config, all_rows),
            "forensic_mode": "laplacian",
        },
    )
    print(json.dumps({
        "train_originals": len(splits["train"]),
        "model_selection_originals": len(splits["model_selection"]),
        "calibration_originals": len(splits["calibration"]),
        "reserved_test_originals": len(splits["test"]),
        "combined_cache": str(args.combined_output),
        "laplacian_cache": str(args.laplacian_output),
    }, indent=2))


if __name__ == "__main__":
    main()
