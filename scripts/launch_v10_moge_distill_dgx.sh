#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${V10_TRAIN_PYTHON:-python}"
MANIFEST="${V10_MANIFEST:-data/celebamask_hq/moge3_base_pilot/manifest.jsonl}"
FEATURE_ROOT="${V10_FEATURE_ROOT:-data/celebamask_hq/moge3_base_features_v10}"
RUN="${V10_RUN:-runs/fastvit_t8_fpn72_v10_moge_feature_kd_384x512}"
LOG="${V10_LOG:-logs/v10_moge_feature_kd_384x512.log}"
PID_FILE="${V10_PID:-logs/v10_moge_feature_kd_384x512.pid}"

test -s "$MANIFEST" || { echo "MoGe manifest missing: $MANIFEST"; exit 1; }
test -s "$FEATURE_ROOT/metadata.json" || { echo "Teacher feature cache missing: $FEATURE_ROOT"; exit 1; }
command -v setsid >/dev/null || { echo "setsid is required on DGX"; exit 1; }
mkdir -p logs
if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "V10 training already running: PID $OLD_PID"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

"$PYTHON" -c 'import torch, timm; assert torch.cuda.is_available(), "CUDA unavailable"'
setsid nohup bash -c '
  "$@" &
  child=$!
  forward_signal() { kill "$child" 2>/dev/null || true; }
  trap forward_signal TERM INT
  wait "$child"
  status=$?
  printf "TRAIN_EXIT_STATUS=%s\n" "$status"
  exit "$status"
' v10_train "$PYTHON" -u scripts/train_v10_moge_distill_dgx.py \
  --manifest "$MANIFEST" --feature-root "$FEATURE_ROOT" --run-dir "$RUN" \
  --height 512 --width 384 --batch-size 4 --epochs 100 \
  --feature-weight 0.02 --workers 4 --log-every 100 --snapshot-interval 10 "$@" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "V10 depth distillation started or resumed: PID $(cat "$PID_FILE")"
echo "Log: $LOG"
echo "Follow: tail -f $LOG"
