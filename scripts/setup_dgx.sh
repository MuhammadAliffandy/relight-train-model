#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${RELIGHT_CONDA_ENV:-relight-dgx}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Load or install Miniconda/Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml" --prune
else
  conda env create -f "$PROJECT_ROOT/environment.yml"
fi

conda run -n "$ENV_NAME" python - <<'PY'
import torch
import torchvision
import timm
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this environment. Check the DGX driver and selected PyTorch wheel.")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
PY

echo "Conda environment '$ENV_NAME' is ready."
echo "Next: run scripts/audit_dgx.sh before training."
