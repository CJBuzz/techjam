from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import load_labeled_paths, stratified_train_val_test_split
from .features import extract_features
from .metrics import classification_metrics
from .model import ExpertMixtureHead, FrozenEncoders, load_checkpoint
from .train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clean and transformed-image robustness")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        dest="checkpoints",
        help="Checkpoint to evaluate; repeat for compatible checkpoints to reuse extracted features",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness.json"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Use validation while comparing candidates; reserve test for the final selected model",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    checkpoint_paths = args.checkpoints or [Path("artifacts/hybrid_detector.pt")]
    models = []
    for checkpoint_path in checkpoint_paths:
        head, config, temperature, _ = load_checkpoint(checkpoint_path, device)
        models.append((str(checkpoint_path), head, config, temperature))
    reference_config = models[0][2]
    encoder_fields = ("clip_model", "clip_dim", "forensic_dim", "forensic_mode")
    reference_signature = tuple(getattr(reference_config, field) for field in encoder_fields)
    if any(tuple(getattr(config, field) for field in encoder_fields) != reference_signature for _, _, config, _ in models[1:]):
        raise ValueError("All checkpoints in one evaluation must use the same encoder configuration")
    encoder_config = reference_config
    if any(config.quality_dim for _, _, config, _ in models):
        from dataclasses import replace
        encoder_config = replace(reference_config, quality_dim=6)
    encoders = FrozenEncoders(encoder_config, device)
    all_rows = load_labeled_paths(args.data_dir)
    _, validation_rows, test_rows = stratified_train_val_test_split(
        all_rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
    )
    rows = validation_rows if args.split == "validation" else test_rows
    print(f"Evaluating untouched {args.split} split: {len(rows)} of {len(all_rows)} source images")
    metadata = {
        "total_source_images": len(all_rows),
        "split": args.split,
        "split_images": len(rows),
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "transform_strengths": "sampled deterministically from the challenge settings per test image",
    }
    model_results = {path: {} for path, _, _, _ in models}
    for name in ("clean", "jpeg", "blur", "resize", "noise", "color", "crop"):
        features, labels, paths = extract_features(rows, encoders, args.batch_size, transform_mode=name)
        sources = [Path(path).parent.name.lower() for path in paths]
        for checkpoint_path, head, _, temperature in models:
            model_config = next(config for path, _, config, _ in models if path == checkpoint_path)
            model_features = features if model_config.quality_dim else features[:, : model_config.clip_dim + model_config.forensic_dim]
            with torch.no_grad():
                probabilities = torch.sigmoid(head(model_features.to(device)) / temperature).cpu()
            by_source = {}
            for source in sorted(set(sources)):
                mask = torch.tensor([item == source for item in sources], dtype=torch.bool)
                by_source[source] = classification_metrics(labels[mask], probabilities[mask])
            model_results[checkpoint_path][name] = {
                "overall": classification_metrics(labels, probabilities), "by_source": by_source
            }
            if isinstance(head, ExpertMixtureHead):
                with torch.no_grad(): gate = head.gate_weights(model_features.to(device)).cpu()
                model_results[checkpoint_path][name]["fft_gate"] = {
                    "overall_mean": float(gate.mean()),
                    "by_source_mean": {
                        source: float(gate[torch.tensor([item == source for item in sources])].mean())
                        for source in sorted(set(sources))
                    },
                }
    if len(models) == 1:
        results = {
            "_metadata": {**metadata, "checkpoint": models[0][0]},
            **model_results[models[0][0]],
        }
    else:
        results = {"_metadata": metadata, "checkpoints": model_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
