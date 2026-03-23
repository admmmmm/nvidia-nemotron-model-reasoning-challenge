#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env.server ]]; then
  # shellcheck disable=SC1091
  source .env.server
fi

ENV_NAME="${ENV_NAME:-nemotron}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please use the Miniconda3 image or install conda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio
python -m pip install \
  transformers \
  accelerate \
  peft \
  bitsandbytes \
  datasets \
  sentencepiece \
  huggingface_hub \
  safetensors \
  pandas \
  scipy

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install || true
fi

python - <<'PY'
import torch
print('python:', __import__('sys').version)
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_name:', torch.cuda.get_device_name(0))
PY
