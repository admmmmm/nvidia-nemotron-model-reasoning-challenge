#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env.server ]]; then
  # shellcheck disable=SC1091
  source .env.server
fi

RUN_NAME="${RUN_NAME:-nemotron_lora_v0}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"
BOOTSTRAP_LOG="$LOG_DIR/${RUN_NAME}_launcher_$(date '+%Y%m%d_%H%M%S').log"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"

nohup bash scripts/run_nemotron_lora.sh >>"$BOOTSTRAP_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "started training pid=$TRAIN_PID log=$BOOTSTRAP_LOG"

nohup bash scripts/monitor_training.sh "$TRAIN_PID" "$RUN_NAME" "$BOOTSTRAP_LOG" >/dev/null 2>&1 &
echo "started monitor for pid=$TRAIN_PID"
