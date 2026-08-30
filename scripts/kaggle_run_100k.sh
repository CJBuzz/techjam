#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 DATA_ROOT [OUTPUT_DIR]" >&2
  exit 2
fi

project_dir=$(cd "$(dirname "$0")/.." && pwd)
data_root=$(cd "$1" && pwd)
output_dir=${2:-/kaggle/working/aigc_100k}
feature_batch_size=${FEATURE_BATCH_SIZE:-32}
head_batch_size=${HEAD_BATCH_SIZE:-256}

mkdir -p "$output_dir"
cd "$project_dir"
python -c 'import torch; assert torch.cuda.is_available(), "Kaggle GPU is not enabled"; x = torch.ones(1, device="cuda"); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list(), x.item())'
python "$project_dir/scripts/kaggle_dataset.py" validate --data-dir "$data_root"

combined_cache="$output_dir/mixed_100k_laplacian_fft_features.pt"
laplacian_cache="$output_dir/mixed_100k_laplacian_features.pt"
laplacian_checkpoint="$output_dir/mixed_100k_laplacian_initializer.pt"
combined_checkpoint="$output_dir/mixed_100k_balanced_consistency_w01.pt"

python "$project_dir/scripts/extract_scale_features.py" \
  --data-dir "$data_root" \
  --split-manifest "$data_root/split_manifest.csv" \
  --combined-output "$combined_cache" \
  --laplacian-output "$laplacian_cache" \
  --augmentation-repeats 3 --batch-size "$feature_batch_size" \
  --seed 42 --device cuda

python -m aigc_detector.train \
  --data-dir "$data_root" --split-manifest "$data_root/split_manifest.csv" \
  --cache "$laplacian_cache" --output "$laplacian_checkpoint" \
  --forensic-mode laplacian --augmentation-policy balanced --augmentation-repeats 3 \
  --modality-dropout 0.1 --head-batch-size "$head_batch_size" \
  --epochs 40 --patience 7 --learning-rate 1e-4 --seed 42 --device cuda

python -m aigc_detector.train \
  --data-dir "$data_root" --split-manifest "$data_root/split_manifest.csv" \
  --cache "$combined_cache" --output "$combined_checkpoint" \
  --forensic-mode laplacian_fft --augmentation-policy balanced --augmentation-repeats 3 \
  --initialize-from-laplacian "$laplacian_checkpoint" --consistency-weight 0.1 \
  --head-batch-size "$head_batch_size" --epochs 40 --patience 7 \
  --learning-rate 1e-4 --seed 42 --device cuda

printf 'Completed without reading reserved test features.\nArtifacts: %s\n' "$output_dir"
