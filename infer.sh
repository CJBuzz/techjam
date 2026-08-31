#!/usr/bin/env bash
set -euo pipefail

# Usage: ./infer.sh [image_dir] [checkpoint] [output_json]
IMAGE_DIR="${1:-images}"
CHECKPOINT="${2:-artifacts/robust_laplacian_fft.pt}"
OUTPUT_JSON="${3:-output.json}"

uv run python scripts/predict_directory.py "$IMAGE_DIR" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_JSON"

echo "Inference results written to $OUTPUT_JSON"
