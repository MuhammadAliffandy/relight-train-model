#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

INIT="runs/fastvit_t8_fpn72_v9_3_face_depth3d_concatall_512/best_checkpoint.pt"
MANIFEST="data/face_depth_3d/manifest.jsonl"
RUN="runs/fastvit_t8_fpn72_v9_4_face_depth3d_smooth100_512"
LOG="logs/v9_4_face_depth3d_smooth100_512.log"
PID_FILE="logs/v9_4_face_depth3d_smooth100_512.pid"

test -f "$INIT" || { echo "Checkpoint V9.3 tidak ditemukan: $INIT"; exit 1; }
test -f "$MANIFEST" || { echo "Manifest FaceDepth tidak ditemukan: $MANIFEST"; exit 1; }
test ! -e "$RUN" || { echo "Run directory sudah ada: $RUN; pilih nama run baru agar hasil lama aman."; exit 1; }
mkdir -p logs
python -c 'from relight_train.model import PortraitGeometryModel'

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Training sudah berjalan dengan PID $OLD_PID. Log: $LOG"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

nohup bash -c '
  python -u scripts/train_face_depth3d_dgx.py "$@" &
  child=$!
  forward_term() { kill "$child" 2>/dev/null || true; }
  trap forward_term TERM INT
  wait "$child"
  status=$?
  printf "TRAIN_EXIT_STATUS=%s\n" "$status"
  exit "$status"
' v9_4 \
  --manifest "$MANIFEST" \
  --init-checkpoint "$INIT" \
  --run-dir "$RUN" \
  --fusion concat_all \
  --size 512 \
  --batch-size 4 \
  --epochs 100 \
  --patience 100 \
  --min-delta 0.00005 \
  --smoothness-weight 0.05 \
  --snapshot-interval 25 \
  --workers 0 \
  --log-every 1 \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "Training V9.4 dimulai. PID: $(cat "$PID_FILE")"
echo "Log: $LOG"
echo "Pantau dengan: tail -f $LOG"
