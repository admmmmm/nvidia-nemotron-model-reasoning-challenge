#!/usr/bin/env bash
set -euo pipefail

PID="${1:?PID required}"
RUN_NAME="${2:?RUN_NAME required}"
LOG_FILE="${3:?LOG_FILE required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env.server ]]; then
  # shellcheck disable=SC1091
  source .env.server
fi

mkdir -p outputs/logs
STATUS_FILE="outputs/logs/${RUN_NAME}_status.json"
PUSH_INTERVAL_SECONDS="${PUSH_INTERVAL_SECONDS:-600}"
SLEEP_SECONDS="${STATUS_SLEEP_SECONDS:-60}"
last_push_epoch=0

write_status() {
  local status="$1"
  local now_iso
  now_iso="$(date -Is)"
  local tail_text
  tail_text="$(tail -n 40 "$LOG_FILE" 2>/dev/null | sed 's/\\/\\\\/g; s/\"/\\"/g')"
  local gpu_state
  gpu_state="$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/\\/\\\\/g; s/\"/\\"/g')"
  local mem_state
  mem_state="$(free -h 2>/dev/null | sed 's/\\/\\\\/g; s/\"/\\"/g')"

  cat > "$STATUS_FILE" <<EOF
{
  "run_name": "$RUN_NAME",
  "pid": $PID,
  "status": "$status",
  "updated_at": "$now_iso",
  "log_file": "$LOG_FILE",
  "gpu": "$gpu_state",
  "memory": "$mem_state",
  "tail": "$tail_text"
}
EOF
}

maybe_push() {
  local now_epoch
  now_epoch="$(date +%s)"
  if [[ "${AUTO_GIT_PUSH:-0}" != "1" ]]; then
    return
  fi
  if (( now_epoch - last_push_epoch < PUSH_INTERVAL_SECONDS )); then
    return
  fi
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return
  fi
  local branch="${GIT_BRANCH:-main}"
  git add "$STATUS_FILE" >/dev/null 2>&1 || true
  if ! git diff --cached --quiet; then
    git commit -m "chore: update training status for $RUN_NAME" >/dev/null 2>&1 || true
    git push origin "$branch" >/dev/null 2>&1 || true
  fi
  last_push_epoch="$now_epoch"
}

while kill -0 "$PID" >/dev/null 2>&1; do
  write_status "running"
  maybe_push
  sleep "$SLEEP_SECONDS"
done

wait_status="finished"
if [[ -f "$LOG_FILE" ]] && grep -q "Training failed" "$LOG_FILE" 2>/dev/null; then
  wait_status="failed"
fi
write_status "$wait_status"
maybe_push
