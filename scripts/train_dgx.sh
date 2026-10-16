#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${RELIGHT_CONDA_ENV:-relight-dgx}"
CONFIG="${1:-$PROJECT_ROOT/configs/dgx_fastvit_t8_depth_normals_50k.yaml}"
LOG_DIR="${RELIGHT_LOG_DIR:-$PROJECT_ROOT/logs}"
RUNTIME_DIR="${RELIGHT_RUNTIME_DIR:-$PROJECT_ROOT/.cache/relight_runtime}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/train_$STAMP.log"

mkdir -p "$LOG_DIR" "$RUNTIME_DIR/huggingface"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUNTIME_DIR/huggingface"

echo "Training log: $LOG_FILE"
echo "To follow from another SSH session: tail -n 100 -f '$LOG_FILE'"
if [[ -r /sys/fs/cgroup/memory.events ]]; then
  echo "Cgroup memory events before training:"
  cat /sys/fs/cgroup/memory.events
fi
cd "$PROJECT_ROOT"
set +e
conda run -n "$ENV_NAME" --no-capture-output python -u -m relight_train.train --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
echo "Training command exit status: $TRAIN_STATUS" | tee -a "$LOG_FILE"
if [[ -r /sys/fs/cgroup/memory.events ]]; then
  echo "Cgroup memory events after exit:" | tee -a "$LOG_FILE"
  cat /sys/fs/cgroup/memory.events | tee -a "$LOG_FILE"
fi
exit "$TRAIN_STATUS"
