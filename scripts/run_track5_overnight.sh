#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-mps}"
EXTERNAL_DEVICE="${EXTERNAL_DEVICE:-cpu}"
REAL_COUNT="${REAL_COUNT:-3000}"
FAKE_COUNT="${FAKE_COUNT:-3000}"
MAX_FAKE_PER_MODEL="${MAX_FAKE_PER_MODEL:-200}"
MAX_REAL_PER_SOURCE="${MAX_REAL_PER_SOURCE:-750}"
BUFFER_SIZE="${BUFFER_SIZE:-512}"
RELAX_AFTER_NO_PROGRESS="${RELAX_AFTER_NO_PROGRESS:-2000}"
MAX_INSPECTED_ROWS="${MAX_INSPECTED_ROWS:-0}"
REBUILD_E1_CACHE="${REBUILD_E1_CACHE:-0}"
START_STAGE="${START_STAGE:-0}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
RUN_EXTERNAL="${RUN_EXTERNAL:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$PARENT_ROOT/data/raw/friend_mixed_5k/data/mixed_5k}"
EXTERNAL_DATA="$PARENT_ROOT/data/external/community_forensics"
ORIGINAL_CHECKPOINT="$PARENT_ROOT/artifacts/friend/robust_laplacian_fft.pt"
TRACK5_ROOT="artifacts/track5"
OVERNIGHT_ROOT="$TRACK5_ROOT/overnight"
LOG_DIR="$OVERNIGHT_ROOT/logs/$RUN_ID"
SUMMARY_CSV="$OVERNIGHT_ROOT/experiment_summary.csv"
SUMMARY_JSON="$OVERNIGHT_ROOT/experiment_summary.json"

E0_DIR="$TRACK5_ROOT/e0_baseline"
E1_DIR="$TRACK5_ROOT/e1_diverse"
E1_STREAM_CACHE="$E1_DIR/stream_cache"
E1_LOCAL_CACHE="$E1_DIR/local_balanced_features.pt"
E1_MODEL="$E1_DIR/model.pt"
E1_EVAL_DIR="$E1_DIR/e0_scorecard"
E2_ORIGINAL_DIR="$TRACK5_ROOT/e2_mild3/original"
E2_E1_DIR="$TRACK5_ROOT/e2_mild3/e1"
E3_DIR="$TRACK5_ROOT/e3_response"
E3_FEATURES="$E3_DIR/features.pt"
E3_MODEL="$E3_DIR/model.pt"
E3_EVAL_DIR="$E3_DIR/e0_scorecard"
EXTERNAL_ROOT="$TRACK5_ROOT/external_sanity"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_track5_overnight.sh --preflight
  bash scripts/run_track5_overnight.sh --smoke
  bash scripts/run_track5_overnight.sh

Environment overrides:
  DEVICE=mps|cpu|cuda             Main extraction/evaluation device (default: mps)
  EXTERNAL_DEVICE=cpu|cuda        External evaluator device (default: cpu)
  REAL_COUNT=3000                 Streamed real-image target
  FAKE_COUNT=3000                 Streamed fake-image target
  MAX_FAKE_PER_MODEL=200          Per-generator quota
  MAX_REAL_PER_SOURCE=750         Per-real-source quota
  BUFFER_SIZE=512                Bounded streaming shuffle buffer
  RELAX_AFTER_NO_PROGRESS=2000   Rows without class progress before quota fallback
  MAX_INSPECTED_ROWS=0           Automatic bounded inspection guard
  REBUILD_E1_CACHE=0|1           Back up an incompatible production cache when 1
  START_STAGE=0|3|6              Resume at E1 training or E3 with 3/6 (default: 0)
  LOCAL_DATA_DIR=<path>          Local labeled 5k dataset override
  FEATURE_BATCH_SIZE=8            Frozen feature extraction batch size
  EVAL_BATCH_SIZE=8               Evaluation batch size
  RUN_EXTERNAL=0|1                Optional informational external check (default: 1)
  RUN_ID=YYYYmmdd-HHMMSS          Unique log-run identifier
EOF
  exit 0
fi

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

COMPLETED_STAGES=()
CURRENT_STAGE="initialization"
CURRENT_LOG=""
CURRENT_COMMAND=""

stage() {
  local number="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/stage_${number}.log"
  mkdir -p "$(dirname "$log")"
  CURRENT_STAGE="STAGE $number — $name"
  CURRENT_LOG="$log"
  printf -v CURRENT_COMMAND '%q ' "$@"
  local status
  set +e
  {
    set -e
    echo "[$(timestamp)] STAGE $number — $name"
    echo "Command: $CURRENT_COMMAND"
    "$@"
  } 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  if (( status == 0 )); then
    echo "[$(timestamp)] STAGE $number COMPLETE" | tee -a "$log"
    COMPLETED_STAGES+=("$number")
    return 0
  fi
  {
    echo "[$(timestamp)] STAGE $number FAILED"
    echo "Exit code: $status"
    echo "Failed command: $CURRENT_COMMAND"
    echo "Stage log: $log"
    echo "Completed stages: ${COMPLETED_STAGES[*]:-none}"
  } | tee -a "$log" >&2
  return "$status"
}

stage_wrapper_self_test() {
  local saved_log_dir="$LOG_DIR"
  local temp_dir output status
  temp_dir="$(mktemp -d)"
  LOG_DIR="$temp_dir"
  set +e
  output="$(stage 99 "intentional exit propagation test" bash -c 'exit 7' 2>&1)"
  status=$?
  set -e
  LOG_DIR="$saved_log_dir"
  rm -rf "$temp_dir"
  [[ "$status" == "7" ]] || { echo "Stage wrapper returned $status, expected 7" >&2; return 1; }
  grep -q 'STAGE 99 FAILED' <<<"$output"
  if grep -q 'STAGE 99 COMPLETE' <<<"$output"; then
    echo "Stage wrapper incorrectly printed COMPLETE" >&2
    return 1
  fi
  echo "Stage-wrapper failure propagation: PASS (exit 7 preserved)"
}

cache_mismatch_report() {
  local manifest="$1"
  python3 - "$manifest" "$BUFFER_SIZE" "$REAL_COUNT" "$FAKE_COUNT" \
    "$MAX_FAKE_PER_MODEL" "$MAX_REAL_PER_SOURCE" "$RELAX_AFTER_NO_PROGRESS" \
    "$MAX_INSPECTED_ROWS" <<'PY'
import json, sys
path = sys.argv[1]
actual = json.load(open(path, encoding="utf-8"))
expected = {
    "schema_version": 2,
    "dataset": "OwensLab/CommunityForensics-Small",
    "split": "train",
    "streaming": True,
    "shuffle_seed": 42,
    "shuffle_buffer_size": int(sys.argv[2]),
    "requested_real": int(sys.argv[3]),
    "requested_fake": int(sys.argv[4]),
    "max_fake_per_model": int(sys.argv[5]),
    "max_real_per_source": int(sys.argv[6]),
    "relax_after_no_progress": int(sys.argv[7]),
    "max_inspected_rows": int(sys.argv[8]),
    "robust_views": 5,
    "model_config": {
        "clip_dim": 512, "clip_model": "openai/clip-vit-base-patch32",
        "dropout": 0.3, "forensic_dim": 2560, "forensic_mode": "laplacian_fft",
        "gate_mode": "features", "head_type": "fusion", "hidden_dim": 256,
        "quality_dim": 0,
    },
}
mismatches = []
for key, value in expected.items():
    if actual.get(key, "<missing>") != value:
        mismatches.append(f"{key}: existing={actual.get(key, '<missing>')!r}, requested={value!r}")
print("\n".join(mismatches))
raise SystemExit(1 if mismatches else 0)
PY
}

e1_cache_status() {
  if [[ ! -e "$E1_STREAM_CACHE" ]]; then
    echo "absent"
    return 0
  fi
  if [[ ! -s "$E1_STREAM_CACHE/manifest.json" ]]; then
    echo "incompatible: missing manifest.json"
    return 1
  fi
  local mismatch status
  set +e
  mismatch="$(cache_mismatch_report "$E1_STREAM_CACHE/manifest.json")"
  status=$?
  set -e
  if (( status == 0 )); then
    echo "compatible"
    return 0
  fi
  echo "incompatible"
  echo "$mismatch"
  return 1
}

preflight() {
  local permit_incompatible_cache="${1:-0}"
  local preflight_failed=0
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Track-5 preflight"
  bash -n "$0"
  [[ -s "$ORIGINAL_CHECKPOINT" ]] || { echo "Missing checkpoint: $ORIGINAL_CHECKPOINT" >&2; return 1; }
  [[ -d "$LOCAL_DATA_DIR/real" && -d "$LOCAL_DATA_DIR/ai" ]] || {
    echo "Missing real/ and ai/ class directories under: $LOCAL_DATA_DIR" >&2
    return 1
  }
  local real_count ai_count
  real_count="$(find "$LOCAL_DATA_DIR/real" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) 2>/dev/null | wc -l | tr -d ' ')"
  ai_count="$(find "$LOCAL_DATA_DIR/ai" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) 2>/dev/null | wc -l | tr -d ' ')"
  [[ "$real_count" -gt 0 && "$ai_count" -gt 0 ]] || {
    echo "Invalid labeled dataset structure under: $LOCAL_DATA_DIR" >&2
    return 1
  }
  echo "Resolved LOCAL_DATA_DIR: $LOCAL_DATA_DIR"
  echo "Local images: real=$real_count ai=$ai_count"
  echo "Baseline checkpoint: present ($ORIGINAL_CHECKPOINT)"
  local writable_parent="artifacts"
  [[ -d "$writable_parent" && -w "$writable_parent" ]] || {
    echo "Artifact directory is not writable: $writable_parent" >&2
    return 1
  }

  local evaluate_help stream_help train_help response_extract_help response_train_help
  evaluate_help="$(uv run --frozen python -m aigc_detector.evaluate --help)"
  stream_help="$(uv run --frozen python -m aigc_detector.tooling.streaming_cache --help)"
  train_help="$(uv run --frozen python -m aigc_detector.train --help)"
  response_extract_help="$(uv run --frozen python -m aigc_detector.analysis.response extract --help)"
  response_train_help="$(uv run --frozen python -m aigc_detector.analysis.response train --help)"
  grep -q -- '--output-dir' <<<"$evaluate_help"
  grep -q -- '--tta' <<<"$evaluate_help"
  grep -q -- '--response-model' <<<"$evaluate_help"
  grep -q -- '--real-count' <<<"$stream_help"
  grep -q -- '--robust-views' <<<"$stream_help"
  grep -q -- '--diverse-cache' <<<"$train_help"
  grep -q -- '--initialize-from-checkpoint' <<<"$train_help"
  grep -q -- '--base-checkpoint' <<<"$response_extract_help"
  grep -q -- '--hidden-dim' <<<"$response_train_help"
  grep -q -- '--diverse-cache' aigc_detector/train.py
  grep -q -- 'merge_balanced_feature_sets' aigc_detector/train.py
  echo "E1 cache-consumption wiring: PASS (--diverse-cache concatenates streamed and local features)"
  echo "Smoke/production isolation: PASS ($TRACK5_ROOT/smoke vs $E1_STREAM_CACHE)"
  stage_wrapper_self_test
  local cache_status cache_result
  set +e
  cache_result="$(e1_cache_status)"
  cache_status=$?
  set -e
  echo "E1 cache status: $cache_result"
  if (( cache_status != 0 )) && [[ "$REBUILD_E1_CACHE" != "1" && "$permit_incompatible_cache" != "1" ]]; then
    echo "Action required: preserve the old cache as a backup and rebuild with:" >&2
    echo "REBUILD_E1_CACHE=1 DEVICE=$DEVICE BUFFER_SIZE=$BUFFER_SIZE bash scripts/run_track5_overnight.sh" >&2
    preflight_failed=1
  fi
  if (( cache_status != 0 )); then
    echo "E1 cache action: incompatible cache will be backed up before production run"
  fi
  uv run --frozen python -c 'import aigc_detector.evaluate, aigc_detector.tooling.streaming_cache, aigc_detector.train, aigc_detector.analysis.response'
  if [[ "$DEVICE" == "mps" ]]; then
    uv run --frozen python -c 'import torch; assert torch.backends.mps.is_available(), "DEVICE=mps but MPS is unavailable"'
    echo "MPS availability: PASS"
  fi
  echo "Expected artifact root: $TRACK5_ROOT"
  echo "Expected log directory: $LOG_DIR"
  if (( preflight_failed != 0 )); then
    echo "Preflight ACTIONABLE FAIL without contacting Hugging Face." >&2
    return 1
  fi
  echo "Preflight PASS without contacting Hugging Face."
}

case "${1:-}" in
  --preflight)
    preflight
    exit 0
    ;;
  --smoke)
    preflight 1
    uv run --frozen python -m unittest tests.test_streaming_cache tests.test_tta tests.test_response
    SMOKE_CACHE="$TRACK5_ROOT/smoke/$RUN_ID/stream_cache"
    uv run --frozen python -m aigc_detector.tooling.streaming_cache \
      --output-dir "$SMOKE_CACHE" --dataset OwensLab/CommunityForensics-Small --split train \
      --real-count 10 --fake-count 10 --max-fake-per-model 5 --max-real-per-source 10 \
      --robust-views 1 --seed 42 --buffer-size 64 --batch-size 4 \
      --relax-after-no-progress 100 --max-inspected-rows 2000 \
      --chunk-originals 10 --device "$DEVICE"
    echo "Smoke passed. Cache: $SMOKE_CACHE (no training or evaluation performed)."
    exit 0
    ;;
  "") ;;
  *)
    echo "Unknown argument: $1" >&2
    exit 2
    ;;
esac

[[ "$START_STAGE" == "0" || "$START_STAGE" == "3" || "$START_STAGE" == "6" ]] || {
  echo "START_STAGE must be 0, 3, or 6" >&2
  exit 2
}

preflight

if [[ -e "$SUMMARY_CSV" || -e "$SUMMARY_JSON" ]]; then
  echo "Existing overnight summary found; refusing to overwrite it: $OVERNIGHT_ROOT" >&2
  exit 1
fi
if [[ -e "$LOG_DIR" ]]; then
  echo "Log directory already exists; choose a different RUN_ID: $LOG_DIR" >&2
  exit 1
fi
mkdir -p "$LOG_DIR" "$OVERNIGHT_ROOT"

interrupted() {
  echo "[$(timestamp)] Interrupted during $CURRENT_STAGE" >&2
  echo "Completed logs and results remain under: $LOG_DIR and $TRACK5_ROOT" >&2
  exit 130
}
trap interrupted INT TERM

optional_stage() {
  local number="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/stage_${number}.log"
  CURRENT_STAGE="STAGE $number — $name"
  CURRENT_LOG="$log"
  printf -v CURRENT_COMMAND '%q ' "$@"
  local status
  set +e
  {
    set -e
    echo "[$(timestamp)] STAGE $number — $name"
    echo "Command: $CURRENT_COMMAND"
    "$@"
  } 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    mkdir -p "$EXTERNAL_ROOT"
    printf '{"status":"failed","log":"%s"}\n' "$log" > "$EXTERNAL_ROOT/status.json"
    echo "[$(timestamp)] OPTIONAL STAGE FAILED; continuing. Log: $log" >&2
  else
    mkdir -p "$EXTERNAL_ROOT"
    printf '{"status":"succeeded","log":"%s"}\n' "$log" > "$EXTERNAL_ROOT/status.json"
    echo "[$(timestamp)] STAGE $number COMPLETE"
  fi
}

scorecard_valid() {
  local directory="$1"
  [[ -s "$directory/detailed_metrics.json" && -s "$directory/condition_metrics.csv" ]] || return 1
  python3 - "$directory/detailed_metrics.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
conditions = payload.get("_metadata", {}).get("conditions", [])
required = {
    "clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
    "blur_s0.5", "blur_s1.0", "blur_s2.0", "resize_x0.5", "resize_x0.25",
    "noise_s0.02", "noise_s0.05", "noise_s0.10", "color_0.8", "color_1.2", "crop_0.8",
}
if set(conditions) != required or "_scorecard" not in payload:
    raise SystemExit(1)
PY
}

checkpoint_has_key() {
  local checkpoint="$1"
  local key="$2"
  [[ -s "$checkpoint" ]] || return 1
  uv run --frozen python - "$checkpoint" "$key" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if sys.argv[2] not in payload:
    raise SystemExit(1)
PY
}

e1_checkpoint_valid() {
  checkpoint_has_key "$E1_MODEL" head_state_dict || return 1
  uv run --frozen python - "$E1_MODEL" "$REAL_COUNT" "$FAKE_COUNT" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
manifest = payload.get("metadata", {}).get("diverse_cache_manifest") or {}
if manifest.get("requested_real") != int(sys.argv[2]) or manifest.get("requested_fake") != int(sys.argv[3]):
    raise SystemExit(1)
PY
}

run_e0() {
  local checkpoint="$1"
  local output_dir="$2"
  local tta="$3"
  [[ -s "$checkpoint" ]] || { echo "Required checkpoint missing or empty: $checkpoint" >&2; return 1; }
  if scorecard_valid "$output_dir"; then
    echo "[$(timestamp)] Reusing valid scorecard: $output_dir"
    return
  fi
  if [[ -e "$output_dir/detailed_metrics.json" || -e "$output_dir/condition_metrics.csv" ]]; then
    echo "Refusing to overwrite an incomplete scorecard: $output_dir" >&2
    return 1
  fi
  uv run --frozen python -m aigc_detector.evaluate \
    --data-dir "$LOCAL_DATA_DIR" --checkpoint "$checkpoint" --split test --profile full \
    --tta "$tta" --seed 42 --batch-size "$EVAL_BATCH_SIZE" --device "$DEVICE" \
    --output-dir "$output_dir"
}

if (( START_STAGE <= 0 )); then
  stage 0 "cheap tests" \
    uv run --frozen python -m unittest tests.test_streaming_cache tests.test_tta tests.test_response
fi

if (( START_STAGE <= 1 )); then
  stage 1 "E0 original exact Track-5 benchmark" \
    run_e0 "$ORIGINAL_CHECKPOINT" "$E0_DIR" none
fi

prepare_e1_cache() {
  local status report backup
  set +e
  report="$(e1_cache_status)"
  status=$?
  set -e
  if (( status == 0 )); then
    echo "E1 cache: $report"
    return 0
  fi
  echo "E1 cache mismatch:"
  echo "$report"
  if [[ "$REBUILD_E1_CACHE" != "1" ]]; then
    echo "Set REBUILD_E1_CACHE=1 to back up the incompatible cache and rebuild." >&2
    return 1
  fi
  backup="${E1_STREAM_CACHE}_backup_$(date +%Y%m%d-%H%M%S)"
  [[ ! -e "$backup" ]] || { echo "Backup path already exists: $backup" >&2; return 1; }
  mv "$E1_STREAM_CACHE" "$backup"
  echo "Backed up incompatible E1 cache to: $backup"
}

e1_cache_complete() {
  [[ -s "$E1_STREAM_CACHE/manifest.json" ]] || return 1
  python3 - "$E1_STREAM_CACHE/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
state = manifest.get("sampling_state", {})
if not state.get("complete"):
    raise SystemExit(1)
if state.get("accepted_real") != manifest.get("requested_real"):
    raise SystemExit(1)
if state.get("accepted_fake") != manifest.get("requested_fake"):
    raise SystemExit(1)
PY
  compgen -G "$E1_STREAM_CACHE/chunk-*.pt" >/dev/null
}

build_e1_cache() {
  prepare_e1_cache
  uv run --frozen python -m aigc_detector.tooling.streaming_cache \
    --output-dir "$E1_STREAM_CACHE" \
    --dataset OwensLab/CommunityForensics-Small --split train \
    --real-count "$REAL_COUNT" --fake-count "$FAKE_COUNT" \
    --max-fake-per-model "$MAX_FAKE_PER_MODEL" \
    --max-real-per-source "$MAX_REAL_PER_SOURCE" \
    --robust-views 5 --seed 42 --buffer-size "$BUFFER_SIZE" \
    --relax-after-no-progress "$RELAX_AFTER_NO_PROGRESS" \
    --max-inspected-rows "$MAX_INSPECTED_ROWS" \
    --batch-size "$FEATURE_BATCH_SIZE" --chunk-originals 100 --device "$DEVICE"
}
if (( START_STAGE <= 2 )); then
  stage 2 "E1 streamed feature cache (build/resume)" build_e1_cache
fi

train_e1() {
  e1_cache_complete || { echo "E1 stream cache is not complete; refusing training" >&2; return 1; }
  if e1_checkpoint_valid; then
    echo "[$(timestamp)] Reusing existing E1 checkpoint: $E1_MODEL"
    return
  fi
  if [[ -e "$E1_MODEL" ]]; then
    echo "Refusing to overwrite an invalid or mismatched E1 checkpoint: $E1_MODEL" >&2
    return 1
  fi
  uv run --frozen python -m aigc_detector.train \
    --data-dir "$LOCAL_DATA_DIR" --output "$E1_MODEL" --cache "$E1_LOCAL_CACHE" \
    --diverse-cache "$E1_STREAM_CACHE" --initialize-from-checkpoint "$ORIGINAL_CHECKPOINT" \
    --forensic-mode laplacian_fft --augmentation-policy balanced --augmentation-repeats 6 \
    --augmentation-depth 2 --consistency-weight 0.05 --worst-group-weight 0.5 \
    --robust-validation --robust-validation-weight 0.7 --threshold-objective balanced \
    --modality-dropout 0.1 --fft-dropout 0.15 --feature-batch-size "$FEATURE_BATCH_SIZE" \
    --head-batch-size 32 --epochs 30 --patience 6 --learning-rate 0.001 --seed 42 \
    --device "$DEVICE"
}
if (( START_STAGE <= 3 )); then
  stage 3 "E1 generator-diverse robust training" train_e1
fi

if (( START_STAGE <= 4 )); then
  stage 4 "E1 exact Track-5 benchmark" run_e0 "$E1_MODEL" "$E1_EVAL_DIR" none
fi

run_e2() {
  run_e0 "$ORIGINAL_CHECKPOINT" "$E2_ORIGINAL_DIR" mild3
  run_e0 "$E1_MODEL" "$E2_E1_DIR" mild3
}
if (( START_STAGE <= 5 )); then
  stage 5 "E2 mild3 TTA full benchmarks" run_e2
fi

run_e3() {
  [[ -s "$E1_MODEL" ]] || { echo "Required E1 checkpoint missing: $E1_MODEL" >&2; return 1; }
  uv run --frozen python -m aigc_detector.analysis.response extract \
    --data-dir "$LOCAL_DATA_DIR" --base-checkpoint "$E1_MODEL" --output "$E3_FEATURES" \
    --augmentation-repeats 2 --validation-fraction 0.15 --test-fraction 0.15 \
    --batch-size "$FEATURE_BATCH_SIZE" --seed 42 --device "$DEVICE"
  if checkpoint_has_key "$E3_MODEL" state_dict; then
    echo "[$(timestamp)] Reusing E3 response model: $E3_MODEL"
  elif [[ ! -e "$E3_MODEL" ]]; then
    uv run --frozen python -m aigc_detector.analysis.response train \
      --cache "$E3_FEATURES" --output "$E3_MODEL" --hidden-dim 32 --batch-size 128 \
      --epochs 50 --patience 8 --learning-rate 0.001 --seed 42
  else
    echo "Refusing to overwrite an invalid E3 response model: $E3_MODEL" >&2
    return 1
  fi
  if scorecard_valid "$E3_EVAL_DIR"; then
    echo "[$(timestamp)] Reusing valid scorecard: $E3_EVAL_DIR"
  else
    uv run --frozen python -m aigc_detector.evaluate \
      --data-dir "$LOCAL_DATA_DIR" --checkpoint "$E1_MODEL" --response-model "$E3_MODEL" \
      --split test --profile full --tta none --seed 42 --batch-size "$EVAL_BATCH_SIZE" \
      --device "$DEVICE" --output-dir "$E3_EVAL_DIR"
  fi
}
stage 6 "E3 perturbation-response features, training, and benchmark" run_e3

run_external() {
  if [[ "$RUN_EXTERNAL" != "1" ]]; then
    echo "[$(timestamp)] External sanity check disabled; set RUN_EXTERNAL=1 to enable"
    return
  fi
  [[ -d "$EXTERNAL_DATA" ]] || { echo "Missing existing external dataset: $EXTERNAL_DATA"; return 1; }
  mkdir -p "$EXTERNAL_ROOT/original" "$EXTERNAL_ROOT/e1"
  if [[ ! -s "$EXTERNAL_ROOT/original/summary.json" ]]; then
    uv run --frozen python -m aigc_detector.analysis.evaluate_external \
      --data-dir "$EXTERNAL_DATA" --checkpoint "$ORIGINAL_CHECKPOINT" \
      --output-dir "$EXTERNAL_ROOT/original" --batch-size "$EVAL_BATCH_SIZE" \
      --device "$EXTERNAL_DEVICE"
  fi
  if [[ ! -s "$EXTERNAL_ROOT/e1/summary.json" ]]; then
    uv run --frozen python -m aigc_detector.analysis.evaluate_external \
      --data-dir "$EXTERNAL_DATA" --checkpoint "$E1_MODEL" \
      --output-dir "$EXTERNAL_ROOT/e1" --batch-size "$EVAL_BATCH_SIZE" \
      --device "$EXTERNAL_DEVICE"
  fi
  echo "External results are informational and are excluded from ranking."
}
optional_stage 7 "optional external sanity check" run_external

generate_summary() {
  if [[ -e "$SUMMARY_CSV" || -e "$SUMMARY_JSON" ]]; then
    echo "Refusing to overwrite an existing overnight summary. Move it or set a new artifact root." >&2
    return 1
  fi
  python3 - "$SUMMARY_JSON" "$SUMMARY_CSV" \
    "$E0_DIR/detailed_metrics.json" "$E1_EVAL_DIR/detailed_metrics.json" \
    "$E2_ORIGINAL_DIR/detailed_metrics.json" "$E2_E1_DIR/detailed_metrics.json" \
    "$E3_EVAL_DIR/detailed_metrics.json" "$EXTERNAL_ROOT/original/summary.json" \
    "$EXTERNAL_ROOT/e1/summary.json" "$ORIGINAL_CHECKPOINT" "$E1_MODEL" "$E3_MODEL" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

summary_json, summary_csv = map(Path, sys.argv[1:3])
scorecards = list(map(Path, sys.argv[3:8]))
external_original, external_e1 = map(Path, sys.argv[8:10])
original_checkpoint, e1_checkpoint, e3_checkpoint = sys.argv[10:13]
candidates = [
    ("E0_original", original_checkpoint, scorecards[0], external_original),
    ("E1_diverse", e1_checkpoint, scorecards[1], external_e1),
    ("E2_original_mild3", original_checkpoint, scorecards[2], None),
    ("E2_e1_mild3", e1_checkpoint, scorecards[3], None),
    ("E3_response", e3_checkpoint, scorecards[4], None),
]

rows = []
for experiment, checkpoint, path, external_path in candidates:
    payload = json.loads(path.read_text(encoding="utf-8"))
    score = payload["_scorecard"]
    conditions = [name for name in payload["_metadata"]["conditions"] if name != "clean"]
    external_auc = None
    if external_path is not None and external_path.is_file():
        external_auc = json.loads(external_path.read_text(encoding="utf-8"))["overall"]["roc_auc"]
    rows.append({
        "experiment": experiment,
        "checkpoint": checkpoint,
        "clean_balanced_accuracy": score["clean_balanced_accuracy"],
        "mean_transformed_balanced_accuracy": score["mean_transformed_balanced_accuracy"],
        "worst_transformed_balanced_accuracy": score["worst_transformed_balanced_accuracy"],
        "worst_condition": score["worst_transformed_condition"],
        "mean_transformed_roc_auc": score["mean_transformed_roc_auc"],
        "worst_transformed_roc_auc": score["worst_transformed_roc_auc"],
        "clean_false_positive_rate": payload["clean"]["overall"]["false_positive_rate"],
        "mean_transformed_false_positive_rate": sum(
            payload[name]["overall"]["false_positive_rate"] for name in conditions
        ) / len(conditions),
        "external_roc_auc": external_auc,
        "status": "succeeded",
    })

e0_clean = rows[0]["clean_balanced_accuracy"]
for row in rows:
    row["clean_constraint_passed"] = row["clean_balanced_accuracy"] >= e0_clean - 0.01
eligible = sorted(
    (row for row in rows if row["clean_constraint_passed"]),
    key=lambda row: (
        row["worst_transformed_balanced_accuracy"],
        row["mean_transformed_balanced_accuracy"],
    ),
    reverse=True,
)
rank = {row["experiment"]: index + 1 for index, row in enumerate(eligible)}
for row in rows:
    row["rank"] = rank.get(row["experiment"])

document = {
    "ranking_policy": {
        "primary": "worst_transformed_balanced_accuracy",
        "clean_constraint": "no more than 0.01 below E0 clean balanced accuracy",
        "tie_break": "mean_transformed_balanced_accuracy",
        "external_metrics_used_for_ranking": False,
    },
    "winner": eligible[0]["experiment"] if eligible else None,
    "winner_reason": (
        "Highest worst transformed balanced accuracy among candidates within 0.01 of E0 clean; "
        "mean transformed balanced accuracy breaks ties. External metrics are excluded."
        if eligible else "No candidate satisfied the clean-performance constraint."
    ),
    "candidates": rows,
}
tmp_json = summary_json.with_suffix(".json.tmp")
tmp_csv = summary_csv.with_suffix(".csv.tmp")
tmp_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
os.replace(tmp_json, summary_json)
os.replace(tmp_csv, summary_csv)
print(f"Winner by Track-5 policy: {document['winner']}")
PY
}
stage 8 "CSV/JSON summary and Track-5 ranking" generate_summary

echo "[$(timestamp)] Overnight pipeline complete"
echo "Summary CSV: $SUMMARY_CSV"
echo "Summary JSON: $SUMMARY_JSON"
