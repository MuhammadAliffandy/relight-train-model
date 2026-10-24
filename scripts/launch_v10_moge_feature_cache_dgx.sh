#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${MOGE_PYTHON:-python}"
MANIFEST="${V10_MANIFEST:-data/celebamask_hq/moge3_base_pilot/manifest.jsonl}"
CHECKPOINT="${MOGE_CHECKPOINT:-models/moge-3-vitl/model.pt}"
OUTPUT="${V10_FEATURE_ROOT:-data/celebamask_hq/moge3_base_features_v10}"
LOG="${V10_FEATURE_LOG:-logs/v10_moge_feature_cache.log}"
PID_FILE="${V10_FEATURE_PID:-logs/v10_moge_feature_cache.pid}"

test -s "$MANIFEST" || { echo "MoGe manifest missing: $MANIFEST"; exit 1; }
test -s "$CHECKPOINT" || { echo "MoGe checkpoint missing: $CHECKPOINT"; exit 1; }
command -v setsid >/dev/null || { echo "setsid is required on DGX"; exit 1; }
mkdir -p logs "$OUTPUT"
if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Feature cache already running: PID $OLD_PID"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

"$PYTHON" scripts/patch_flexgemm_triton32_dgx.py
"$PYTHON" -c 'import torch; from moge.model.v3 import MoGeModel; assert torch.cuda.is_available(), "CUDA unavailable"'
setsid nohup bash -c '
  "$@" &
  child=$!
  forward_signal() { kill "$child" 2>/dev/null || true; }
  trap forward_signal TERM INT
  wait "$child"
  status=$?
  printf "FEATURE_EXIT_STATUS=%s\n" "$status"
  exit "$status"
' v10_feature "$PYTHON" -u scripts/cache_moge3_teacher_features_dgx.py \
  --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --output "$OUTPUT" "$@" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "MoGe feature cache started: PID $(cat "$PID_FILE")"
echo "Log: $LOG"
echo "Follow: tail -f $LOG"
