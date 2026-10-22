#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${MOGE_PYTHON:-python}"
IMAGES="${MOGE_IMAGES:-data/celebamask_hq/images}"
CHECKPOINT="${MOGE_CHECKPOINT:-models/moge-3-vitl/model.pt}"
OUTPUT="${MOGE_OUTPUT:-data/celebamask_hq/moge3_pseudo}"
LOG="${MOGE_LOG:-logs/moge3_celebamask_hq_pseudolabel.log}"
PID_FILE="${MOGE_PID_FILE:-logs/moge3_celebamask_hq_pseudolabel.pid}"

test -d "$IMAGES" || { echo "Image directory missing: $IMAGES"; exit 1; }
test -s "$CHECKPOINT" || { echo "MoGe checkpoint missing: $CHECKPOINT"; exit 1; }
command -v setsid >/dev/null || { echo "setsid is required on DGX"; exit 1; }
mkdir -p "$(dirname "$LOG")" "$(dirname "$PID_FILE")" "$OUTPUT"

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "MoGe pseudo-label service already running: PID $OLD_PID"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

"$PYTHON" scripts/patch_flexgemm_triton32_dgx.py
"$PYTHON" -c 'import torch; from moge.model.v3 import MoGeModel; assert torch.cuda.is_available(), "CUDA unavailable"'

setsid nohup "$PYTHON" -u scripts/generate_moge3_pseudo_depth_dgx.py \
  --images "$IMAGES" --checkpoint "$CHECKPOINT" --output "$OUTPUT" "$@" \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "MoGe pseudo-label service started: PID $(cat "$PID_FILE")"
echo "Log: $LOG"
echo "Follow: tail -f $LOG"
