#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env.server ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.server
  set +a
fi

RUN_NAME="${RUN_NAME:-nemotron_lora_v0}"
RUN_FOLDER_NAME="${RUN_FOLDER_NAME:-${1:-}}"
if [[ -z "$RUN_FOLDER_NAME" ]]; then
  RUN_FOLDER_NAME="${RUN_NAME}_$(date '+%Y%m%d_%H%M')"
fi

GIT_SYNC_SOURCE_REPO="${GIT_SYNC_SOURCE_REPO:-$REPO_ROOT}"
GIT_SYNC_REPO="${GIT_SYNC_REPO:-/root/nemotron_git}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-60}"
START_SYNC_LOOP="${START_SYNC_LOOP:-1}"

mkdir -p outputs/logs

pkill -f "python -m src.train.sft_local" >/dev/null 2>&1 || true
pkill -f "bash scripts/run_nemotron_lora.sh" >/dev/null 2>&1 || true

if [[ "$START_SYNC_LOOP" == "1" ]]; then
  pkill -f "sync_loop.sh $GIT_SYNC_SOURCE_REPO $GIT_SYNC_REPO $RUN_NAME" >/dev/null 2>&1 || true
  setsid nohup bash "$REPO_ROOT/scripts/sync_loop.sh" \
    "$GIT_SYNC_SOURCE_REPO" \
    "$GIT_SYNC_REPO" \
    "$RUN_NAME" \
    "$SYNC_INTERVAL_SECONDS" \
    > "$REPO_ROOT/outputs/logs/sync_loop.out" 2>&1 < /dev/null &
  echo $! > "$REPO_ROOT/outputs/logs/sync_loop.pid"
fi

setsid nohup env RUN_FOLDER_NAME="$RUN_FOLDER_NAME" bash "$REPO_ROOT/scripts/run_nemotron_lora.sh" \
  > "$REPO_ROOT/outputs/logs/launch_latest.out" 2>&1 < /dev/null &
echo $! > "$REPO_ROOT/outputs/logs/launch_latest.pid"

echo "run_name=$RUN_NAME"
echo "run_folder_name=$RUN_FOLDER_NAME"
echo "launch_pid=$(<"$REPO_ROOT/outputs/logs/launch_latest.pid")"
if [[ -f "$REPO_ROOT/outputs/logs/sync_loop.pid" ]]; then
  echo "sync_loop_pid=$(<"$REPO_ROOT/outputs/logs/sync_loop.pid")"
fi
echo "launch_log=$REPO_ROOT/outputs/logs/launch_latest.out"
