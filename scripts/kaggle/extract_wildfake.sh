#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 IMAGE_ROOT METADATA_CSV OUTPUT_DIR [TOTAL_ORIGINALS]" >&2
  exit 2
fi

project_dir=$(cd "$(dirname "$0")/../.." && pwd)
image_root=$(cd "$1" && pwd)
metadata=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
output_dir=$3
total_originals=${4:-100000}

cd "$project_dir"
python -c 'import torch; assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator"; print(torch.__version__, torch.cuda.get_device_name(0))'
python scripts/kaggle/extract_wildfake.py \
  --image-root "$image_root" --metadata "$metadata" --output-dir "$output_dir" \
  --total-originals "$total_originals" --views-per-image 3 \
  --batch-size "${FEATURE_BATCH_SIZE:-32}" --chunk-originals "${CHUNK_ORIGINALS:-64}" \
  --seed "${SEED:-42}" --device cuda

printf 'Raw-image-free feature cache ready under %s\n' "$output_dir"
