#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${RELIGHT_CONDA_ENV:-relight-dgx}"
CONFIG="${1:-$PROJECT_ROOT/configs/dgx_fastvit_t8_depth_normals_local30000_seed42.yaml}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

conda run -n "$ENV_NAME" --no-capture-output python -u -m relight_train.stage_local_subset --config "$CONFIG"
