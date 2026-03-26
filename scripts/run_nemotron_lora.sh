#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env.server ]]; then
  # shellcheck disable=SC1091
  set -a
  source .env.server
  set +a
fi

ENV_NAME="${ENV_NAME:-nemotron}"
RUN_NAME="${RUN_NAME:-nemotron_lora_v0}"
MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
TRAIN_FILE="${TRAIN_FILE:-train.csv}"
SPLIT_DIR="${SPLIT_DIR:-data/splits/default}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/adapters/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_FOLDER_NAME="${RUN_FOLDER_NAME:-}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_MEMORY_GPU="${MAX_MEMORY_GPU:-39GiB}"
MAX_MEMORY_CPU="${MAX_MEMORY_CPU:-8GiB}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
FORCE_FULL_GPU="${FORCE_FULL_GPU:-1}"

RUN_TS_MINUTE="$(date '+%Y%m%d_%H%M')"
if [[ -z "$RUN_FOLDER_NAME" ]]; then
  RUN_FOLDER_NAME="${RUN_NAME}_${RUN_TS_MINUTE}"
fi
RUN_LOG_DIR="${LOG_DIR}/${RUN_FOLDER_NAME}"
LOG_FILE="${RUN_LOG_DIR}/train_${RUN_TS_MINUTE}.log"

mkdir -p "$LOG_DIR" "$RUN_LOG_DIR" "$OUTPUT_DIR"
echo "$RUN_LOG_DIR" > "${LOG_DIR}/${RUN_NAME}_latest_log_dir.txt"
echo "$LOG_FILE" > "${LOG_DIR}/${RUN_NAME}_latest_log_path.txt"

if [[ "$MODEL_NAME" == *"Nemotron"* ]] || [[ "$MODEL_NAME" == *"nemotron"* ]]; then
  # Nemotron on this host repeatedly OOM-killed under CPU offload path.
  FORCE_FULL_GPU=1
fi

if ! command -v conda >/dev/null 2>&1; then
  for conda_root in \
    "/home/vipuser/anaconda3" \
    "$HOME/anaconda3" \
    "/opt/conda" \
    "/usr/local/miniconda3"
  do
    if [[ -f "$conda_root/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_root/etc/profile.d/conda.sh"
      export PATH="$conda_root/bin:$PATH"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please use the Miniconda3 image or install conda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('huggingface login ok')
PY
fi

if [[ ! -f "$SPLIT_DIR/train.jsonl" ]] || [[ ! -f "$SPLIT_DIR/val.jsonl" ]]; then
  python -m src.data.preprocess --train-file "$TRAIN_FILE" --output-dir "$SPLIT_DIR"
fi

cmd=(
  python -m src.train.sft_local
  --model-name "$MODEL_NAME"
  --train-file "$SPLIT_DIR/train.jsonl"
  --val-file "$SPLIT_DIR/val.jsonl"
  --output-dir "$OUTPUT_DIR"
  --max-length "$MAX_LENGTH"
  --num-train-epochs "$NUM_EPOCHS"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --logging-steps "$LOGGING_STEPS"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --max-memory-gpu "$MAX_MEMORY_GPU"
  --max-memory-cpu "$MAX_MEMORY_CPU"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  cmd+=(--local-files-only)
fi
if [[ "$FORCE_FULL_GPU" == "1" ]]; then
  cmd+=(--force-full-gpu)
fi

if [[ -n "$TRAIN_LIMIT" ]]; then
  cmd+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$VAL_LIMIT" ]]; then
  cmd+=(--val-limit "$VAL_LIMIT")
fi

echo "=== effective runtime config ==="
echo "RUN_NAME=$RUN_NAME"
echo "RUN_FOLDER_NAME=$RUN_FOLDER_NAME"
echo "MODEL_NAME=$MODEL_NAME"
echo "MAX_LENGTH=$MAX_LENGTH"
echo "BATCH(train/eval)=$PER_DEVICE_TRAIN_BATCH_SIZE/$PER_DEVICE_EVAL_BATCH_SIZE"
echo "GRAD_ACC=$GRADIENT_ACCUMULATION_STEPS"
echo "MAX_MEMORY_GPU=$MAX_MEMORY_GPU"
echo "MAX_MEMORY_CPU=$MAX_MEMORY_CPU"
echo "LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"
echo "FORCE_FULL_GPU=$FORCE_FULL_GPU"
echo "RUN_LOG_DIR=$RUN_LOG_DIR"
echo "LOG_FILE=$LOG_FILE"

set +e
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
train_exit=${PIPESTATUS[0]}
set -e

if [[ "$train_exit" -eq 0 ]]; then
  bash scripts/finish_run.sh SUCCESS "$RUN_NAME" "$LOG_FILE" "$OUTPUT_DIR" "$RUN_LOG_DIR"
  echo "Training finished successfully. Log: $LOG_FILE"
else
  bash scripts/finish_run.sh FAILED "$RUN_NAME" "$LOG_FILE" "$OUTPUT_DIR" "$RUN_LOG_DIR"
  echo "Training failed. Log: $LOG_FILE" >&2
  exit "$train_exit"
fi
