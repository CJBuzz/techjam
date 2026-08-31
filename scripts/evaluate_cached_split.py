from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from aigc_detector.data import image_source, load_split_manifest
from aigc_detector.metrics import classification_metrics
from aigc_detector.model import load_checkpoint
from aigc_detector.train import choose_device


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def score(
    checkpoint: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    sources: list[str],
    device: torch.device,
) -> dict:
    head, config, temperature, metadata = load_checkpoint(checkpoint, device)
    expected_width = config.clip_dim + config.forensic_dim + config.quality_dim
    if features.shape[1] != expected_width:
        raise ValueError(
            f"Cache width {features.shape[1]} does not match {checkpoint} width {expected_width}"
        )
    with torch.inference_mode():
        probabilities = torch.sigmoid(head(features.to(device)) / temperature).cpu()
    threshold = float(metadata.get("threshold", 0.5))
    per_source = {}
    for source in sorted(set(sources)):
        mask = torch.tensor([value == source for value in sources])
        per_source[source] = classification_metrics(
            labels[mask], probabilities[mask], threshold
        )
    held_out_mask = torch.tensor([
        source in {"wildfake_adm_holdout", "wildfake_imagenet"} for source in sources
    ])
    legacy_mask = ~held_out_mask
    return {
        "checkpoint": str(checkpoint.resolve()),
        "temperature": temperature,
        "threshold": threshold,
        "overall": classification_metrics(labels, probabilities, threshold),
        "held_out_adm_vs_imagenet": classification_metrics(
            labels[held_out_mask], probabilities[held_out_mask], threshold
        ),
        "legacy_model_selection": classification_metrics(
            labels[legacy_mask], probabilities[legacy_mask], threshold
        ),
        "per_source": per_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare compatible checkpoints on one cached, manifest-audited split"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("model_selection", "calibration"), default="model_selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    prefix = "val" if args.split == "model_selection" else "calibration"
    features = cache[f"{prefix}_features"]
    labels = cache[f"{prefix}_labels"]
    rows = load_split_manifest(args.data_dir, args.split_manifest)[args.split]
    expected_labels = torch.tensor([label for _, label in rows], dtype=torch.float32)
    if len(rows) != len(labels) or not torch.equal(labels, expected_labels):
        raise ValueError("Cache labels do not align with the requested manifest split")
    sources = [image_source(path, args.data_dir) for path, _ in rows]
    device = choose_device(args.device)
    report = {
        "data_dir": str(args.data_dir.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "feature_cache": str(args.feature_cache.resolve()),
        "split": args.split,
        "selection_rule": "ROC-AUC and average precision; threshold metrics are diagnostic only",
        "models": [score(path, features, labels, sources, device) for path in args.checkpoint],
    }
    report = json_safe(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
