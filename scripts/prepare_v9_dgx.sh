#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${RELIGHT_CONDA_ENV:-relight-dgx}"
SAMPLES="${V9_REAL_SAMPLES:-1000}"
V8_CHECKPOINT="$PROJECT_ROOT/runs/fastvit_t8_fpn72_v8_moderate_crop_50k_seed42_384x512_30ep/best_checkpoint.pt"
V9_CONFIG="$PROJECT_ROOT/configs/dgx_fastvit_t8_depth_normals_local5000_seed42_v9_da2_distill_pilot.yaml"

run_env() {
  if [[ ${CONDA_DEFAULT_ENV:-} == "$ENV_NAME" ]]; then
    python "$@"
  else
    conda run -n "$ENV_NAME" --no-capture-output python "$@"
  fi
}

cd "$PROJECT_ROOT"
if [[ ! -f "$V8_CHECKPOINT" ]]; then
  echo "V8 initialization checkpoint is missing: $V8_CHECKPOINT" >&2
  echo "Copy/restore the V8 best_checkpoint.pt to this path before preparing V9." >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/relight_runtime/huggingface}"
mkdir -p "$HF_HOME" "$PROJECT_ROOT/logs"

echo "Staging the exact 5,000-sample SynthHuman subset required by V9"
"$PROJECT_ROOT/scripts/stage_dgx_subset.sh" "$V9_CONFIG"

echo "Installing V9 data and teacher dependencies in $ENV_NAME"
run_env -m pip install "fiftyone>=1.0,<2" "transformers>=4.45,<5" safetensors

if [[ ! -f "$PROJECT_ROOT/data/v9_openimages_people/manifest.jsonl" ]]; then
  echo "Downloading up to $SAMPLES Open Images V7 person samples"
  run_env scripts/download_v9_openimages_people.py \
    --max-samples "$SAMPLES" \
    --output "$PROJECT_ROOT/data/v9_openimages_people"
else
  echo "Reusing existing Open Images manifest"
fi

echo "Creating deterministic 80/20 real-image distillation split"
run_env scripts/split_v9_real_manifest.py \
  --manifest "$PROJECT_ROOT/data/v9_openimages_people/manifest.jsonl" \
  --validation-fraction 0.2 \
  --seed 42

echo "Generating frozen Depth Anything V2 teacher maps"
run_env scripts/generate_v9_pseudo_depth.py \
  --manifest "$PROJECT_ROOT/data/v9_openimages_people/manifest.jsonl" \
  --output "$PROJECT_ROOT/data/v9_openimages_people/pseudo_depth"

echo "V9 input data is ready. Review the source metadata and then start training with:"
echo "./scripts/train_dgx.sh configs/dgx_fastvit_t8_depth_normals_local5000_seed42_v9_da2_distill_pilot.yaml"
