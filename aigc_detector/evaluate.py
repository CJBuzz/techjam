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
from .model import FrozenEncoders, load_checkpoint
from .train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clean and transformed-image robustness")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/hybrid_detector.pt"))
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
    head, config, temperature, _ = load_checkpoint(args.checkpoint, device)
    encoders = FrozenEncoders(config, device)
    all_rows = load_labeled_paths(args.data_dir)
    _, validation_rows, test_rows = stratified_train_val_test_split(
        all_rows, args.data_dir, args.validation_fraction, args.test_fraction, args.seed
    )
    rows = validation_rows if args.split == "validation" else test_rows
    print(f"Evaluating untouched {args.split} split: {len(rows)} of {len(all_rows)} source images")
    results = {
        "_metadata": {
            "total_source_images": len(all_rows),
            "split": args.split,
            "split_images": len(rows),
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "transform_strengths": "sampled deterministically from the challenge settings per test image",
        }
    }
    for name in ("clean", "jpeg", "blur", "resize", "noise", "color", "crop"):
        features, labels, paths = extract_features(rows, encoders, args.batch_size, transform_mode=name)
        with torch.no_grad():
            probabilities = torch.sigmoid(head(features.to(device)) / temperature).cpu()
        sources = [Path(path).parent.name.lower() for path in paths]
        by_source = {}
        for source in sorted(set(sources)):
            mask = torch.tensor([item == source for item in sources], dtype=torch.bool)
            by_source[source] = classification_metrics(labels[mask], probabilities[mask])
        results[name] = {"overall": classification_metrics(labels, probabilities), "by_source": by_source}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
