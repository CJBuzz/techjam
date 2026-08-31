"""Evaluate a locked checkpoint on the local WildFake demonstration corpus.

This intentionally handles the supplied COCO/DALL-E directory layout directly rather
than making it look like training data.  It never applies an additional transform.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image

from aigc_detector.features import extract_features
from aigc_detector.metrics import classification_metrics, expected_calibration_error
from aigc_detector.model import FrozenEncoders, load_checkpoint
from aigc_detector.train import choose_device


def decoded_digest(path: Path) -> tuple[str, tuple[int, int]]:
    """Return a content digest of decoded RGB pixels, not JPEG container bytes."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest(), rgb.size


def select_paths(root: Path, count: int, seed: int) -> list[Path]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})
    if len(paths) < count:
        raise ValueError(f"{root} has {len(paths)} JPEGs; need {count}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    return sorted(paths[:count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked external WildFake COCO/DALL-E evaluation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/wildfake_robust"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0, help="First selected row for a resumable feature chunk")
    parser.add_argument("--end", type=int, default=None, help="Exclusive selected row for a resumable feature chunk")
    parser.add_argument("--feature-output", type=Path, default=None, help="Write one clean feature chunk and stop")
    parser.add_argument("--feature-input", type=Path, action="append", default=[], help="Previously extracted feature chunk")
    args = parser.parse_args()

    real_paths = select_paths(args.data_dir / "Real", args.per_class, args.seed)
    fake_paths = select_paths(args.data_dir / "Diffusion_based", args.per_class, args.seed + 1)
    rows = [(path, 0) for path in real_paths] + [(path, 1) for path in fake_paths]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    digests: dict[str, list[str]] = {}
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "source", "width", "height", "decoded_sha256"))
        writer.writeheader()
        for path, label in rows:
            digest, (width, height) = decoded_digest(path)
            digests.setdefault(digest, []).append(str(path))
            writer.writerow({
                "path": str(path), "label": label,
                "source": "coco2017_val" if label == 0 else "dalle3_advanced",
                "width": width, "height": height, "decoded_sha256": digest,
            })
    duplicate_groups = [paths for paths in digests.values() if len(paths) > 1]
    if duplicate_groups:
        raise ValueError(f"Selected set has {len(duplicate_groups)} decoded-pixel duplicate groups; refusing to score it")

    device = choose_device(args.device)
    head, config, temperature, checkpoint_metadata = load_checkpoint(args.checkpoint, device)
    if args.feature_input:
        chunks = [torch.load(path, map_location="cpu", weights_only=True) for path in args.feature_input]
        features = torch.cat([chunk["features"] for chunk in chunks])
        labels = torch.cat([chunk["labels"] for chunk in chunks])
        paths = [path for chunk in chunks for path in chunk["paths"]]
    else:
        end = len(rows) if args.end is None else args.end
        selected_rows = rows[args.start : end]
        if not selected_rows:
            raise ValueError("Selected feature chunk is empty")
        encoders = FrozenEncoders(config, device)
        features, labels, paths = extract_features(selected_rows, encoders, args.batch_size)  # clean: uploaded pixels only
        if args.feature_output:
            args.feature_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features, "labels": labels, "paths": paths}, args.feature_output)
            print(f"Wrote clean feature chunk {args.start}:{end} to {args.feature_output}")
            return
    with torch.no_grad():
        probabilities = torch.sigmoid(head(features.to(device)) / temperature).cpu()
    predictions = probabilities >= 0.5
    sources = labels == 0
    by_source = {
        "coco2017_val_real": classification_metrics(labels[sources], probabilities[sources]),
        "dalle3_advanced_fake": classification_metrics(labels[~sources], probabilities[~sources]),
    }
    labels_bool = labels.bool()
    report = {
        "scope": "External labelled stress check only; not used for training, selection, calibration, or threshold tuning.",
        "checkpoint": str(args.checkpoint), "checkpoint_metadata": checkpoint_metadata,
        "temperature": temperature, "seed": args.seed, "uploaded_pixels_only": True,
        "selection": {"coco2017_val_real": len(real_paths), "dalle3_advanced_fake": len(fake_paths)},
        "manifest": str(args.manifest), "decoded_pixel_duplicate_groups": 0,
        "overall": classification_metrics(labels, probabilities),
        "expected_calibration_error_15_bins": expected_calibration_error(labels, probabilities),
        "by_source": by_source,
        "confusion_matrix_threshold_0_5": {
            "tn": int((~predictions & ~labels_bool).sum()), "fp": int((predictions & ~labels_bool).sum()),
            "fn": int((~predictions & labels_bool).sum()), "tp": int((predictions & labels_bool).sum()),
        },
        "prediction_summary": {
            "mean_ai_probability": float(probabilities.mean()),
            "median_ai_probability": float(probabilities.median()),
            "real_mean_ai_probability": float(probabilities[sources].mean()),
            "fake_mean_ai_probability": float(probabilities[~sources].mean()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
