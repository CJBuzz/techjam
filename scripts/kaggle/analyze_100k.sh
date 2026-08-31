#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 DATA_ROOT [OUTPUT_DIR]" >&2
  exit 2
fi

project_dir=$(cd "$(dirname "$0")/../.." && pwd)
data_root=$(cd "$1" && pwd)
output_dir=${2:-/kaggle/working/aigc_100k}
feature_batch_size=${FEATURE_BATCH_SIZE:-32}

cache="$output_dir/mixed_100k_laplacian_fft_features.pt"
baseline="$output_dir/mixed_100k_balanced_consistency_w01.pt"
calibrated="$output_dir/mixed_100k_balanced_consistency_w01_mixed_calibrated.pt"

for required in "$cache" "$baseline" "$data_root/split_manifest.csv"; do
  if [[ ! -f "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 1
  fi
done

cd "$project_dir"
python -m aigc_detector.calibrate \
  --data-dir "$data_root" --split-manifest "$data_root/split_manifest.csv" \
  --checkpoint "$baseline" --feature-cache "$cache" \
  --output-checkpoint "$calibrated" \
  --output-report "$output_dir/mixed_100k_calibration.json" \
  --selection mixed --batch-size "$feature_batch_size" --seed 42 --device cuda

python -m aigc_detector.analysis.shortcut_audit \
  --data-dir "$data_root" --split-manifest "$data_root/split_manifest.csv" \
  --feature-cache "$cache" --checkpoint "$calibrated" \
  --output "$output_dir/mixed_100k_shortcut_audit.json" --seed 42 --device cuda

python -m aigc_detector.severity \
  --data-dir "$data_root" --split-manifest "$data_root/split_manifest.csv" \
  --checkpoint "$calibrated" --split model_selection \
  --output "$output_dir/mixed_100k_model_selection_severity.json" \
  --batch-size "$feature_batch_size" --seed 42 --device cuda --resume

printf 'Completed calibration, shortcut audit, and model-selection severity matrix. Reserved test images remain unread.\n'
