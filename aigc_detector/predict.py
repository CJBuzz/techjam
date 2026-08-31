from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .data import find_images
from .model import FrozenEncoders, load_checkpoint
from .train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write calibrated AIGC probabilities for an image directory")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/hybrid_detector.pt"))
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    head, config, temperature, _ = load_checkpoint(args.checkpoint, device)
    encoders = FrozenEncoders(config, device)
    paths = find_images(args.image_dir)
    if not paths:
        raise ValueError(f"No supported images found in {args.image_dir}")
    records = []
    for start in tqdm(range(0, len(paths), args.batch_size), desc="predict"):
        batch_paths = []
        images = []
        for path in paths[start : start + args.batch_size]:
            try:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
                batch_paths.append(path)
            except (OSError, ValueError) as error:
                print(f"Skipping unreadable image {path}: {error}")

        if not images:
            continue
        features = encoders(images).to(device)
        with torch.no_grad():
            probabilities = torch.sigmoid(head(features) / temperature).cpu().tolist()
        records.extend(
            {"image_path": str(path), "pred": float(pred)}
            for path, pred in zip(batch_paths, probabilities, strict=True)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} predictions to {args.output}")


if __name__ == "__main__":
    main()

