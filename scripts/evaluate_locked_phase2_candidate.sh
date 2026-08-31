#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" == "--help" ]]; then
  echo "DEVICE=mps NUM_WORKERS=4 bash scripts/evaluate_locked_phase2_candidate.sh"
  exit 0
fi

DEVICE="${DEVICE:-mps}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PARENT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
DATA_DIR="${LOCAL_DATA_DIR:-$PARENT_ROOT/data/raw/friend_mixed_5k/data/mixed_5k}"
TRACK5_ROOT="${TRACK5_ROOT:-artifacts/track5}"
RECOMMENDATION="$TRACK5_ROOT/phase2/recommended_candidate.json"
OUTPUT_DIR="$TRACK5_ROOT/phase2/final_locked_test"

[[ -f "$RECOMMENDATION" ]] || { echo "Missing recommendation: $RECOMMENDATION" >&2; exit 2; }
mapfile_output="$({ .venv/bin/python - "$RECOMMENDATION" <<'PY'
import json, sys
document = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"experiment", "checkpoint_config_paths", "selection_split", "final_test_evaluated", "clean_constraint_pass"}
if not required <= document.keys():
    raise SystemExit("Incomplete Phase-2 recommendation")
if document["selection_split"] != "validation" or document["final_test_evaluated"] is not False:
    raise SystemExit("Recommendation is not a validation-only locked candidate")
if document["clean_constraint_pass"] is not True or len(document["checkpoint_config_paths"]) != 1:
    raise SystemExit("Recommendation is not an eligible single locked configuration")
print(document["experiment"])
print(document["checkpoint_config_paths"][0])
PY
} 2>&1)" || { echo "$mapfile_output" >&2; exit 2; }
experiment="$(printf '%s\n' "$mapfile_output" | sed -n '1p')"
artifact="$(printf '%s\n' "$mapfile_output" | sed -n '2p')"
[[ -s "$artifact" ]] || { echo "Locked artifact is missing: $artifact" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

case "$experiment" in
  E1b)
    .venv/bin/python -m aigc_detector.evaluate --data-dir "$DATA_DIR" --checkpoint "$artifact" \
      --split test --profile full --tta none --output-dir "$OUTPUT_DIR" --device "$DEVICE"
    ;;
  E1c)
    .venv/bin/python -m aigc_detector.e1c locked-test --data-dir "$DATA_DIR" --locked-ensemble "$artifact" \
      --output-dir "$OUTPUT_DIR" --device "$DEVICE"
    ;;
  E2b)
    .venv/bin/python -m aigc_detector.e2b locked-test --data-dir "$DATA_DIR" --locked-policy "$artifact" \
      --output-dir "$OUTPUT_DIR" --device "$DEVICE"
    ;;
  E5_raw|E5_mild3)
    .venv/bin/python -m aigc_detector.e5 locked-test --data-dir "$DATA_DIR" --locked-calibration "$artifact" \
      --output-dir "$OUTPUT_DIR" --device "$DEVICE"
    ;;
  E4b_fixed|E4b_global|E4b_adaptive)
    .venv/bin/python -m aigc_detector.e4b locked-test --data-dir "$DATA_DIR" --model "$artifact" \
      --output-dir "$OUTPUT_DIR" --device "$DEVICE"
    ;;
  E6)
    .venv/bin/python -m aigc_detector.e6 locked-test --data-dir "$DATA_DIR" --winning-config "$artifact" \
      --output-dir "$OUTPUT_DIR" --device "$DEVICE"
    ;;
  E7)
    .venv/bin/python -m aigc_detector.e7 locked-test --data-dir "$DATA_DIR" --winning-config "$artifact" \
      --output-dir "$OUTPUT_DIR" --device "$DEVICE" --workers "$NUM_WORKERS"
    ;;
  *) echo "Unsupported locked Phase-2 candidate: $experiment" >&2; exit 2 ;;
esac
