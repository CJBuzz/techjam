#!/usr/bin/env bash
set -euo pipefail

# Usage: ./infer.sh [image_dir] [checkpoint] [output_json]
IMAGE_DIR="${1:-images}"
CHECKPOINT="${2:-artifacts/diverse_initialized_40k_calibrated.pt}"
OUTPUT_JSON="${3:-output.json}"

# The existing environment is sufficient for inference; avoid an unnecessary rebuild.
if command -v uv >/dev/null 2>&1; then
  UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}" \
    uv run --no-sync python -m aigc_detector.predict \
      "$IMAGE_DIR" --checkpoint "$CHECKPOINT" --output "$OUTPUT_JSON"
elif [[ -x .venv/bin/python ]]; then
  .venv/bin/python -m aigc_detector.predict \
    "$IMAGE_DIR" --checkpoint "$CHECKPOINT" --output "$OUTPUT_JSON"
else
  echo "Install uv and run 'uv sync' before inference." >&2
  exit 1
fi
echo "Inference results written to $OUTPUT_JSON"
