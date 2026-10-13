#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${RELIGHT_LOG_DIR:-$HOME/raid/Clara/relight_logs}"
LOG_FILE="${1:-}"
if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$(find "$LOG_DIR" -maxdepth 1 -type f -name 'train_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
  echo "No training log found in $LOG_DIR. Start training with scripts/train_dgx.sh first." >&2
  exit 1
fi
echo "Following: $LOG_FILE"
tail -n 100 -f "$LOG_FILE"
