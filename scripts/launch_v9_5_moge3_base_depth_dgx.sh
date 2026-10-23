#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

INIT="runs/fastvit_t8_fpn72_v9_4_face_depth3d_smooth100_512/best_checkpoint.pt"
MANIFEST="data/celebamask_hq/moge3_base_pilot/manifest.jsonl"
RUN="runs/fastvit_t8_fpn72_v9_5_moge3_base_depth_384x512"
LOG="logs/v9_5_moge3_base_depth_384x512.log"
PID_FILE="logs/v9_5_moge3_base_depth_384x512.pid"

test -f "$INIT" || { echo "Checkpoint V9.4 missing: $INIT"; exit 1; }
test -f "$MANIFEST" || { echo "MoGe manifest missing: $MANIFEST"; exit 1; }
command -v setsid >/dev/null || { echo "setsid is required on DGX"; exit 1; }
mkdir -p logs

OLD_RUN_PID_FILE="logs/v9_5_moge3_base_depth_512.pid"
if [ -f "$OLD_RUN_PID_FILE" ]; then
  OLD_RUN_PID="$(cat "$OLD_RUN_PID_FILE")"
  if kill -0 "$OLD_RUN_PID" 2>/dev/null; then
    echo "Stop the previous 512x512 run first: PID $OLD_RUN_PID"
    exit 1
  fi
fi

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Training already running: PID $OLD_PID; log=$LOG"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

setsid nohup bash -c '
  python -u scripts/train_moge3_depth_dgx.py "$@" &
  child=$!
  forward_signal() { kill "$child" 2>/dev/null || true; }
  trap forward_signal TERM INT
  wait "$child"
  status=$?
  printf "TRAIN_EXIT_STATUS=%s\n" "$status"
  exit "$status"
' v9_5 \
  --manifest "$MANIFEST" \
  --init-checkpoint "$INIT" \
  --run-dir "$RUN" \
  --height 512 \
  --width 384 \
  --batch-size 4 \
  --epochs 100 \
  --snapshot-interval 10 \
  --workers 4 \
  --log-every 100 \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "MoGe depth training started or resumed: PID $(cat "$PID_FILE")"
echo "Log: $LOG"
echo "Follow: tail -f $LOG"
