#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${RELIGHT_CONDA_ENV:-relight-dgx}"
CONFIG="${1:-$PROJECT_ROOT/configs/dgx_fastvit_t8_depth_normals_local30000_seed42.yaml}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

RCLONE_BIN="${RELIGHT_RCLONE_BIN:-$(command -v rclone || true)}"
if [[ -z "$RCLONE_BIN" ]] && command -v conda >/dev/null 2>&1; then
  RCLONE_BIN="$(conda run -n base --no-capture-output python -c 'import shutil; print(shutil.which("rclone") or "")' 2>/dev/null || true)"
fi
if [[ -z "$RCLONE_BIN" ]]; then
  echo "rclone was not found in this shell or Conda base. Install it or set RELIGHT_RCLONE_BIN to its full path." >&2
  exit 1
fi
export RELIGHT_RCLONE_BIN="$RCLONE_BIN"
echo "Using rclone executable: $RELIGHT_RCLONE_BIN"

conda run -n "$ENV_NAME" --no-capture-output python -u -m relight_train.stage_local_subset --config "$CONFIG"
