#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:-offline_bundle/mamba_py310_torch211_cu118}"
ENV_NAME="${ENV_NAME:-nemotron_mamba}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WHEEL_DIR="$REPO_ROOT/$BUNDLE_DIR/wheels"
SRC_DIR="$REPO_ROOT/$BUNDLE_DIR/src"
REQ_FILE="$REPO_ROOT/$BUNDLE_DIR/manifests/requirements-offline.txt"

if [[ ! -d "$WHEEL_DIR" ]]; then
  echo "offline wheel dir not found: $WHEEL_DIR" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Use a Miniconda image or install conda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "$WHEEL_DIR"/torch-2.1.1+cu118-cp310-cp310-linux_x86_64.whl
python -m pip install "$WHEEL_DIR"/torchvision-0.16.1+cu118-cp310-cp310-linux_x86_64.whl
python -m pip install "$WHEEL_DIR"/torchaudio-2.1.1+cu118-cp310-cp310-linux_x86_64.whl
python -m pip install --no-index --find-links "$WHEEL_DIR" -r "$REQ_FILE"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if ls "$WHEEL_DIR"/causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl >/dev/null 2>&1; then
  python -m pip install "$WHEEL_DIR"/causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

if ls "$WHEEL_DIR"/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl >/dev/null 2>&1; then
  python -m pip install "$WHEEL_DIR"/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
elif ls "$SRC_DIR"/v1.1.1.zip >/dev/null 2>&1; then
  python -m pip install "$SRC_DIR"/v1.1.1.zip
else
  echo "mamba package not found in bundle" >&2
  exit 1
fi

python - <<'PY'
import torch
import mamba_ssm
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("mamba_ssm:", getattr(mamba_ssm, "__version__", "unknown"))
PY
