from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .data import find_images
from .ensemble import compatible_encoder_config, score_head
from .model import FrozenEncoders, load_checkpoint
from .train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a frozen two-checkpoint ensemble policy")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    policy_document = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = policy_document["policy"]
    device = choose_device(args.device)
    loaded = [
        load_checkpoint(policy["checkpoint_40k"], device),
        load_checkpoint(policy["checkpoint_100k"], device),
    ]
    encoders = FrozenEncoders(compatible_encoder_config([item[1] for item in loaded]), device)
    paths = find_images(args.image_dir)
    if not paths:
        raise ValueError(f"No supported images found in {args.image_dir}")
    records = []
    for start in tqdm(range(0, len(paths), args.batch_size), desc="ensemble predict"):
        batch_paths = paths[start : start + args.batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        features = encoders(images).cpu()
        probabilities = [
            score_head(features, head, config, temperature, device)
            for head, config, temperature, _ in loaded
        ]
        blended = policy["weight_40k"] * probabilities[0] + policy["weight_100k"] * probabilities[1]
        records.extend({"image_path": str(path), "pred": float(prediction)} for path, prediction in zip(batch_paths, blended))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} predictions to {args.output}; decision threshold={policy['threshold']:.8f}")


if __name__ == "__main__":
    main()
