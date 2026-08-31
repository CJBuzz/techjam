#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Usage: ./infer.sh [image_dir] [checkpoint] [output_json]
IMAGE_DIR="${1:-images}"
CHECKPOINT="${2:-artifacts/diverse_initialized_40k_calibrated.pt}"
OUTPUT_JSON="${3:-output.json}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "See README.md for checkpoint setup or pass one as the second argument." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  # On a fresh clone this creates the locked environment. The prefetch command
  # is idempotent: cached weights are reused without downloading them again.
  UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}" \
    uv run --frozen python scripts/download_pretrained_models.py --checkpoint "$CHECKPOINT"
  UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}" \
    uv run --no-sync python -m aigc_detector.predict \
      "$IMAGE_DIR" --checkpoint "$CHECKPOINT" --output "$OUTPUT_JSON"
elif [[ -x .venv/bin/python ]]; then
  .venv/bin/python scripts/download_pretrained_models.py --checkpoint "$CHECKPOINT"
  .venv/bin/python -m aigc_detector.predict \
    "$IMAGE_DIR" --checkpoint "$CHECKPOINT" --output "$OUTPUT_JSON"
else
  echo "Install uv and run 'uv sync --frozen' before inference." >&2
  exit 1
fi

echo "Inference results written to $OUTPUT_JSON"
