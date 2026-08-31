#!/usr/bin/env python3
"""
Batch inference script for testing multiple images with optional ensemble predictions.
Useful for evaluation and testing workflows outside the Streamlit UI.

Usage:
    uv run python scripts/batch_inference.py path/to/images --checkpoint artifacts/model.pt --output results.json --ensemble
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from aigc_detector.data import find_images
from aigc_detector.model import FrozenEncoders, load_checkpoint
from aigc_detector.train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch inference with optional ensemble predictions"
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing images to predict")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        default=[Path("artifacts/hybrid_detector.pt")],
        help="Path(s) to model checkpoint(s). If multiple, will ensemble.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("batch_predictions.json"), help="Output JSON file"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device to use"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Average predictions from all checkpoints (only used if multiple checkpoints provided)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold (0-1)"
    )

    args = parser.parse_args()

    # Validate arguments
    device = choose_device(args.device)
    paths = find_images(args.image_dir)
    if not paths:
        raise ValueError(f"No supported images found in {args.image_dir}")

    print(f"Found {len(paths)} images in {args.image_dir}")

    # Load models
    models = []
    for checkpoint_path in args.checkpoint:
        if not checkpoint_path.exists():
            print(f"⚠️  Warning: Checkpoint not found: {checkpoint_path}")
            continue
        try:
            head, config, temperature, _ = load_checkpoint(checkpoint_path, device)
            encoders = FrozenEncoders(config, device)
            models.append(
                {
                    "name": checkpoint_path.stem,
                    "head": head,
                    "encoders": encoders,
                    "temperature": temperature,
                }
            )
            print(f"✅ Loaded model: {checkpoint_path.stem}")
        except Exception as e:
            print(f"❌ Failed to load {checkpoint_path}: {e}")

    if not models:
        raise ValueError("No models could be loaded")

    # Run inference
    records = []
    for start in tqdm(range(0, len(paths), args.batch_size), desc="Predicting"):
        batch_paths = paths[start : start + args.batch_size]
        images = []
        for path in batch_paths:
            try:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
                continue

        if not images:
            continue

        # Batch inference for each model
        batch_size = len(images)
        predictions_per_model = []

        for model_dict in models:
            try:
                features = model_dict["encoders"](images).to(device)
                with torch.no_grad():
                    logits = model_dict["head"](features)
                    probs = torch.sigmoid(logits / model_dict["temperature"]).cpu().numpy()
                predictions_per_model.append(probs)
            except Exception as e:
                print(f"⚠️  Error during inference with {model_dict['name']}: {e}")
                predictions_per_model.append(np.full(batch_size, np.nan))

        # Process results
        for i, path in enumerate(batch_paths):
            result = {
                "image_path": str(path),
                "image_name": path.name,
                "models": {},
            }

            # Individual model predictions
            for model_idx, model_dict in enumerate(models):
                if predictions_per_model[model_idx] is not None:
                    pred = float(predictions_per_model[model_idx][i])
                    result["models"][model_dict["name"]] = {
                        "probability": pred,
                        "prediction": "AI-generated" if pred > args.threshold else "Real",
                        "confidence": abs(pred - 0.5) * 2,
                    }

            # Ensemble prediction
            valid_predictions = [
                predictions_per_model[j][i]
                for j in range(len(predictions_per_model))
                if not np.isnan(predictions_per_model[j][i])
            ]

            if len(valid_predictions) > 1 and args.ensemble:
                ensemble_pred = float(np.mean(valid_predictions))
                result["ensemble"] = {
                    "probability": ensemble_pred,
                    "prediction": "AI-generated" if ensemble_pred > args.threshold else "Real",
                    "confidence": abs(ensemble_pred - 0.5) * 2,
                    "num_models": len(valid_predictions),
                }
            elif len(valid_predictions) == 1:
                # Single model case
                pred = valid_predictions[0]
                result["ensemble"] = {
                    "probability": float(pred),
                    "prediction": "AI-generated" if pred > args.threshold else "Real",
                    "confidence": abs(float(pred) - 0.5) * 2,
                    "num_models": 1,
                }

            records.append(result)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n✅ Wrote {len(records)} predictions to {args.output}")

    # Print summary statistics
    if records:
        total = len(records)
        ai_count = sum(1 for r in records if r.get("ensemble", {}).get("prediction") == "AI-generated")
        real_count = total - ai_count

        print(f"\nSummary:")
        print(f"  Total images: {total}")
        print(f"  AI-generated: {ai_count} ({100*ai_count/total:.1f}%)")
        print(f"  Real images: {real_count} ({100*real_count/total:.1f}%)")

        # Average confidence
        confidences = [r.get("ensemble", {}).get("confidence", 0) for r in records]
        if confidences:
            print(f"  Average confidence: {np.mean(confidences):.2%}")


if __name__ == "__main__":
    main()
