#!/usr/bin/env bash
set -euo pipefail

cd /home/sj/techjam
until [[ -s data/mixed_100k/split_manifest.csv && -s data/mixed_100k/audit.json ]]; do
  sleep 60
done

export HF_HOME=/home/sj/techjam/.hf-cache
export TORCH_HOME=/home/sj/techjam/.torch-cache
export HF_HUB_OFFLINE=1
exec .venv/bin/python scripts/extract_scale_features.py \
  --data-dir data/mixed_100k \
  --split-manifest data/mixed_100k/split_manifest.csv \
  --combined-output artifacts/mixed_100k_laplacian_fft_features.pt \
  --laplacian-output artifacts/mixed_100k_laplacian_features.pt \
  --augmentation-repeats 3 --batch-size 16 --seed 42 --device cpu
