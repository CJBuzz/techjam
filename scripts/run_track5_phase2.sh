#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" == "--help" ]]; then
  echo "START_STAGE=0 END_STAGE=10 FORCE_STAGE=<optional> DEVICE=mps NUM_WORKERS=4 bash scripts/run_track5_phase2.sh"
  exit 0
fi

START_STAGE="${START_STAGE:-0}"
END_STAGE="${END_STAGE:-10}"
FORCE_STAGE="${FORCE_STAGE:-}"
DEVICE="${DEVICE:-mps}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-8}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-64}"

for value in "$START_STAGE" "$END_STAGE" "$NUM_WORKERS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Stage/worker controls must be non-negative integers" >&2; exit 2; }
done
(( START_STAGE <= END_STAGE && END_STAGE <= 10 )) || { echo "Require 0 <= START_STAGE <= END_STAGE <= 10" >&2; exit 2; }
if [[ -n "$FORCE_STAGE" && ! "$FORCE_STAGE" =~ ^([0-9]|10)$ ]]; then
  echo "FORCE_STAGE must be an integer from 0 through 10" >&2; exit 2
fi

PARENT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
DATA_DIR="${LOCAL_DATA_DIR:-$PARENT_ROOT/data/raw/friend_mixed_5k/data/mixed_5k}"
E0_CHECKPOINT="${E0_CHECKPOINT:-$PARENT_ROOT/artifacts/friend/robust_laplacian_fft.pt}"
TRACK5_ROOT="${TRACK5_ROOT:-artifacts/track5}"
LOCAL_CACHE="${LOCAL_CACHE:-$TRACK5_ROOT/e1_diverse/local_balanced_features.pt}"
STREAM_CACHE="${STREAM_CACHE:-$TRACK5_ROOT/e1_diverse/stream_cache}"
E1_CHECKPOINT="${E1_CHECKPOINT:-$TRACK5_ROOT/e1_diverse/model.pt}"
VALIDATION_CACHE="$TRACK5_ROOT/e4a_ablation/validation_features.pt"
PHASE2_ROOT="$TRACK5_ROOT/phase2"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$PHASE2_ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

COMPLETED_STAGES=()
CURRENT_COMMAND=""

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

json_valid() {
  local path="$1" stage="$2"
  [[ -f "$path" ]] && .venv/bin/python -m aigc_detector.experiments.phase2 validate-artifact --path "$path" --stage "$stage" >/dev/null 2>&1
}

e7_completion_valid() {
  local summary="$TRACK5_ROOT/e7_radial_frequency/validation_summary.json"
  json_valid "$summary" 9 || return 1
  .venv/bin/python - "$summary" "$TRACK5_ROOT/e7_radial_frequency/winning_config.json" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
winner = summary.get("eligible_winner")
no_winner = summary.get("no_eligible_candidate") is True
if winner is None and no_winner and summary.get("reason"):
    raise SystemExit(0)
if winner is not None and pathlib.Path(sys.argv[2]).is_file():
    raise SystemExit(0)
raise SystemExit(1)
PY
}

stage_complete() {
  local stage="$1"
  local override="PHASE2_STAGE_${stage}_ARTIFACT"
  if [[ -n "${!override:-}" ]]; then json_valid "${!override}" "$stage"; return; fi
  case "$stage" in
    1) json_valid "$TRACK5_ROOT/e4a_ablation/ablation_summary.json" 1 ;;
    2) json_valid "$TRACK5_ROOT/e1b_weight_sweep/weight_sweep_summary.json" 2 ;;
    3) json_valid "$TRACK5_ROOT/e1c_ensemble/ensemble_summary.json" 3 && [[ -s "$TRACK5_ROOT/e1c_ensemble/winning_ensemble.json" ]] ;;
    4) json_valid "$TRACK5_ROOT/e2b_tta_search/tta_policy_summary.json" 4 && [[ -s "$TRACK5_ROOT/e2b_tta_search/winning_policy.json" ]] ;;
    5) json_valid "$TRACK5_ROOT/e5_quality_calibration/raw/calibration_summary.json" 5 &&
       json_valid "$TRACK5_ROOT/e5_quality_calibration/mild3/calibration_summary.json" 5 &&
       [[ -s "$TRACK5_ROOT/e5_quality_calibration/raw/winning_calibration.json" && -s "$TRACK5_ROOT/e5_quality_calibration/mild3/winning_calibration.json" ]] ;;
    6) for mode in fixed global adaptive; do json_valid "$TRACK5_ROOT/e4b_adaptive_fusion/$mode/summary.json" 6 && [[ -s "$TRACK5_ROOT/e4b_adaptive_fusion/$mode/model.pt" ]] || return 1; done ;;
    7) json_valid "$TRACK5_ROOT/e4c_gate_intervention/intervention_summary.json" 7 ;;
    8) json_valid "$TRACK5_ROOT/e6_scale_consistency/validation_summary.json" 8 && [[ -s "$TRACK5_ROOT/e6_scale_consistency/winning_config.json" ]] ;;
    9) e7_completion_valid ;;
    10) json_valid "$PHASE2_ROOT/recommended_candidate.json" 10 && [[ -s "$PHASE2_ROOT/phase2_summary.json" ]] ;;
    *) return 1 ;;
  esac
}

stage_0() {
  [[ "$(git branch --show-current)" == "track5-experiments" ]] || { echo "Expected branch track5-experiments" >&2; return 2; }
  [[ -d "$DATA_DIR" && -f "$E0_CHECKPOINT" && -f "$LOCAL_CACHE" && -d "$STREAM_CACHE" && -f "$E1_CHECKPOINT" ]] || {
    echo "Preflight failed: one or more required local Phase-1 inputs are absent" >&2; return 2;
  }
  bash -n scripts/run_track5_phase2.sh scripts/evaluate_locked_phase2_candidate.sh || return
  .venv/bin/python -m unittest tests.test_phase2 tests.test_e1b tests.test_e1c tests.test_e2b tests.test_e4a tests.test_e4b tests.test_e4c tests.test_e5 tests.test_e6 tests.test_e7
}

stage_1() {
  .venv/bin/python -m aigc_detector.e4a --data-dir "$DATA_DIR" --base-cache "$LOCAL_CACHE" \
    --validation-cache "$VALIDATION_CACHE" --output-dir "$TRACK5_ROOT/e4a_ablation" \
    --device "$DEVICE" --feature-batch-size "$FEATURE_BATCH_SIZE" --head-batch-size "$HEAD_BATCH_SIZE" --seed "$SEED"
}
stage_2() {
  .venv/bin/python -m aigc_detector.e1b sweep --local-cache "$LOCAL_CACHE" --external-cache "$STREAM_CACHE" \
    --initialize-from-checkpoint "$E0_CHECKPOINT" --output-dir "$TRACK5_ROOT/e1b_weight_sweep" \
    --device "$DEVICE" --batch-size "$HEAD_BATCH_SIZE" --seed "$SEED"
}
stage_3() {
  .venv/bin/python -m aigc_detector.e1c search --data-dir "$DATA_DIR" --e0-checkpoint "$E0_CHECKPOINT" \
    --e1-checkpoint "$E1_CHECKPOINT" --output-dir "$TRACK5_ROOT/e1c_ensemble" \
    --device "$DEVICE" --batch-size "$FEATURE_BATCH_SIZE" --seed "$SEED"
}
stage_4() {
  .venv/bin/python -m aigc_detector.e2b search --data-dir "$DATA_DIR" --checkpoint "$E0_CHECKPOINT" \
    --output-dir "$TRACK5_ROOT/e2b_tta_search" --device "$DEVICE" --batch-size "$FEATURE_BATCH_SIZE" --seed "$SEED"
}
stage_5() {
  local mode
  for mode in raw mild3; do
    .venv/bin/python -m aigc_detector.e5 validation --data-dir "$DATA_DIR" --checkpoint "$E0_CHECKPOINT" \
      --base-mode "$mode" --output-dir "$TRACK5_ROOT/e5_quality_calibration/$mode" \
      --device "$DEVICE" --batch-size "$FEATURE_BATCH_SIZE" --seed "$SEED" || return
  done
}
stage_6() {
  local mode
  for mode in fixed global adaptive; do
    .venv/bin/python -m aigc_detector.e4b train --data-dir "$DATA_DIR" --base-cache "$LOCAL_CACHE" \
      --validation-cache "$VALIDATION_CACHE" --baseline-checkpoint "$E0_CHECKPOINT" \
      --output-dir "$TRACK5_ROOT/e4b_adaptive_fusion/$mode" --mode "$mode" --device "$DEVICE" \
      --feature-batch-size "$FEATURE_BATCH_SIZE" --batch-size "$HEAD_BATCH_SIZE" --seed "$SEED" || return
  done
}
stage_7() {
  .venv/bin/python -m aigc_detector.e4c --data-dir "$DATA_DIR" --base-cache "$LOCAL_CACHE" \
    --validation-cache "$VALIDATION_CACHE" --model "$TRACK5_ROOT/e4b_adaptive_fusion/adaptive/model.pt" \
    --output-dir "$TRACK5_ROOT/e4c_gate_intervention" --device "$DEVICE" --feature-batch-size "$FEATURE_BATCH_SIZE"
}
stage_8() {
  .venv/bin/python -m aigc_detector.e6 cache --data-dir "$DATA_DIR" --base-cache "$LOCAL_CACHE" \
    --output-dir "$TRACK5_ROOT/e6_scale_consistency/scale_features" --device "$DEVICE" \
    --batch-size "$FEATURE_BATCH_SIZE" --workers "$NUM_WORKERS" || return
  .venv/bin/python -m aigc_detector.e6 sweep --base-cache "$LOCAL_CACHE" --validation-cache "$VALIDATION_CACHE" \
    --scale-cache "$TRACK5_ROOT/e6_scale_consistency/scale_features" --initialize-from-checkpoint "$E0_CHECKPOINT" \
    --output-dir "$TRACK5_ROOT/e6_scale_consistency" --consistency-mode logit_asymmetric \
    --device "$DEVICE" --batch-size "$HEAD_BATCH_SIZE" --seed "$SEED"
}
stage_9() {
  NUM_WORKERS="$NUM_WORKERS" .venv/bin/python -m aigc_detector.e7 cache --data-dir "$DATA_DIR" \
    --base-cache "$LOCAL_CACHE" --output-dir "$TRACK5_ROOT/e7_radial_frequency/features" \
    --bins 16 32 64 --batch-size 32 --workers "$NUM_WORKERS" --seed "$SEED" || return
  .venv/bin/python -m aigc_detector.e7 sweep --data-dir "$DATA_DIR" --base-cache "$LOCAL_CACHE" \
    --validation-cache "$VALIDATION_CACHE" --feature-cache "$TRACK5_ROOT/e7_radial_frequency/features" \
    --baseline-checkpoint "$E0_CHECKPOINT" --output-dir "$TRACK5_ROOT/e7_radial_frequency" \
    --forensic-scale-cache "$TRACK5_ROOT/e6_scale_consistency/scale_features" --bins 16 32 64 \
    --modes radial_only fused_radial clip_radial --device "$DEVICE" \
    --feature-batch-size "$FEATURE_BATCH_SIZE" --batch-size "$HEAD_BATCH_SIZE" --seed "$SEED"
}
stage_10() { .venv/bin/python -m aigc_detector.experiments.phase2 aggregate --track5-root "$TRACK5_ROOT"; }

run_stage() {
  local stage="$1" title="$2" log="$LOG_DIR/stage_${stage}.log"
  if (( stage < START_STAGE || stage > END_STAGE )); then return 0; fi
  if [[ "$FORCE_STAGE" != "$stage" ]] && stage_complete "$stage"; then
    echo "[$(timestamp)] STAGE $stage SKIPPED (validated completion artifact): $title"
    COMPLETED_STAGES+=("$stage:reused")
    return 0
  fi
  local override="PHASE2_STAGE_${stage}_COMMAND"
  CURRENT_COMMAND="${!override:-stage_${stage}}"
  echo "[$(timestamp)] STAGE $stage START: $title"
  echo "Command: $CURRENT_COMMAND" | tee "$log"
  if [[ -n "${!override:-}" ]]; then
    bash -c "${!override}" 2>&1 | tee -a "$log"
  else
    "stage_${stage}" 2>&1 | tee -a "$log"
  fi
  local code="${PIPESTATUS[0]}"
  if (( code != 0 )); then
    echo "STAGE $stage FAILED"
    echo "Exit code: $code"
    echo "Failed command: $CURRENT_COMMAND"
    echo "Stage log: $log"
    echo "Completed stages: ${COMPLETED_STAGES[*]:-none}"
    return "$code"
  fi
  if (( stage > 0 )) && ! stage_complete "$stage"; then
    echo "STAGE $stage FAILED"
    echo "Exit code: 3"
    echo "Failed command: $CURRENT_COMMAND (completion artifact invalid or missing)"
    echo "Stage log: $log"
    echo "Completed stages: ${COMPLETED_STAGES[*]:-none}"
    return 3
  fi
  COMPLETED_STAGES+=("$stage")
  echo "[$(timestamp)] STAGE $stage COMPLETE: $title"
}

STAGE_TITLES=(
  "preflight and cheap tests" "E4a modality ablations" "E1b external-weight sweep"
  "E1c calibrated ensemble" "E2b validation TTA search" "E5 quality calibration"
  "E4b adaptive fusion" "E4c gate interventions" "E6 scale consistency"
  "E7 radial frequency" "aggregate validation results"
)

for stage in $(seq "$START_STAGE" "$END_STAGE"); do
  run_stage "$stage" "${STAGE_TITLES[$stage]}" || exit $?
done
echo "[$(timestamp)] PHASE-2 RUN COMPLETE"
echo "Summary: $PHASE2_ROOT/phase2_summary.json"
