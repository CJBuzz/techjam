from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aigc_detector.data import ROBUST_SELECTION_CONDITIONS, load_labeled_paths, load_split_manifest
from aigc_detector.features import extract_balanced_features, extract_condition_features, extract_features
from aigc_detector.model import FrozenEncoders, ModelConfig
from aigc_detector.train import build_cache_manifest, choose_device


def save_cache(path: Path, train_x: torch.Tensor, train_y: torch.Tensor,
               val_x: torch.Tensor, val_y: torch.Tensor,
               calibration_x: torch.Tensor, calibration_y: torch.Tensor,
               robust_val_x: torch.Tensor, robust_val_y: torch.Tensor,
               robust_val_conditions: list[str],
               groups: list[str], original_indices: torch.Tensor,
               manifest: dict, split_manifest: Path, forensic_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
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
        "test_features_extracted": False,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 100K scale caches without reading the reserved test split")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--laplacian-output", type=Path, required=True)
    parser.add_argument("--augmentation-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    splits = load_split_manifest(args.data_dir, args.split_manifest)
    all_rows = load_labeled_paths(args.data_dir)
    device = choose_device(args.device)
    config = ModelConfig(forensic_mode="laplacian_fft", forensic_dim=2560)
    encoders = FrozenEncoders(config, device)
    train_x, train_y, _, groups, original_indices = extract_balanced_features(
        splits["train"], encoders, args.batch_size, args.augmentation_repeats, args.seed
    )
    val_x, val_y, _ = extract_features(splits["model_selection"], encoders, args.batch_size)
    calibration_x, calibration_y, _ = extract_features(splits["calibration"], encoders, args.batch_size)
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
    combined_manifest = build_cache_manifest(cache_args, config, all_rows)
    save_cache(args.combined_output, train_x, train_y, val_x, val_y,
               calibration_x, calibration_y, robust_val_x, robust_val_y,
               robust_val_conditions, groups, original_indices,
               combined_manifest, args.split_manifest, "laplacian_fft")
    laplacian_width = config.clip_dim + 1280
    laplacian_config = ModelConfig(forensic_mode="laplacian", forensic_dim=1280)
    laplacian_manifest = build_cache_manifest(cache_args, laplacian_config, all_rows)
    save_cache(args.laplacian_output, train_x[:, :laplacian_width], train_y,
               val_x[:, :laplacian_width], val_y,
               calibration_x[:, :laplacian_width], calibration_y,
               robust_val_x[:, :laplacian_width], robust_val_y,
               robust_val_conditions, groups, original_indices,
               laplacian_manifest, args.split_manifest, "laplacian")
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
