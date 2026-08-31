#!/usr/bin/env python3
"""Download the frozen pretrained encoders required by submission inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def prepare_clip(clip_model: str, cache: Path) -> None:
    """Prefer a complete local cache, falling back to an online download."""
    try:
        AutoImageProcessor.from_pretrained(
            clip_model, cache_dir=str(cache), local_files_only=True
        )
        CLIPVisionModelWithProjection.from_pretrained(
            clip_model, cache_dir=str(cache), local_files_only=True
        )
        print("CLIP weights found in the project-local cache")
    except OSError:
        print("CLIP weights are not cached; downloading from Hugging Face")
        AutoImageProcessor.from_pretrained(clip_model, cache_dir=str(cache))
        CLIPVisionModelWithProjection.from_pretrained(clip_model, cache_dir=str(cache))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CLIP ViT-B/32 and EfficientNet-B0 weights for AIGC inference"
    )
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Read the CLIP model identifier from this detector checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_model = args.clip_model
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        clip_model = payload.get("config", {}).get("clip_model", clip_model)
    hf_cache = PROJECT_ROOT / ".hf-cache" / "hub"
    torch_cache = PROJECT_ROOT / ".torch-cache" / "hub"
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)

    print(f"Preparing CLIP processor and weights: {clip_model}")
    prepare_clip(clip_model, hf_cache)

    print("Preparing ImageNet EfficientNet-B0 weights")
    torch.hub.set_dir(str(torch_cache))
    efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    print("Pretrained encoder weights are ready:")
    print(f"  Hugging Face: {hf_cache}")
    print(f"  Torch Hub:    {torch_cache}")


if __name__ == "__main__":
    main()
