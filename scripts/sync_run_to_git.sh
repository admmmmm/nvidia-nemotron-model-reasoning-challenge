#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${1:-/root/nemotron}"
GIT_REPO="${2:-/root/nemotron_git}"
RUN_NAME="${3:-nemotron_lora_v0}"

if [[ ! -d "$SOURCE_REPO" ]]; then
  echo "source repo not found: $SOURCE_REPO" >&2
  exit 1
fi

if [[ ! -d "$GIT_REPO/.git" ]]; then
  echo "git repo not found: $GIT_REPO" >&2
  exit 1
fi

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
  fi
}

copy_dir_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -d "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    cp -a "$src" "$dst"
  fi
}

cd "$GIT_REPO"

branch="$(git branch --show-current 2>/dev/null || true)"
if [[ -z "$branch" ]]; then
  branch="master"
fi

echo "[$(date '+%F %T')] sync_run_to_git: fetch origin/$branch"
git fetch origin "$branch" || true

if git rev-parse --verify "origin/$branch" >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] sync_run_to_git: rebasing onto origin/$branch"
  if ! git rebase "origin/$branch"; then
    echo "[$(date '+%F %T')] sync_run_to_git: rebase failed, aborting rebase and skipping this cycle" >&2
    git rebase --abort >/dev/null 2>&1 || true
    exit 0
  fi
fi

cd "$SOURCE_REPO"

mkdir -p "$GIT_REPO/outputs/logs"

copy_if_exists \
  "$SOURCE_REPO/outputs/logs/launch_latest.out" \
  "$GIT_REPO/outputs/logs/launch_latest.out"
copy_if_exists \
  "$SOURCE_REPO/outputs/logs/launch_latest.pid" \
  "$GIT_REPO/outputs/logs/launch_latest.pid"
copy_if_exists \
  "$SOURCE_REPO/outputs/logs/${RUN_NAME}_latest_log_dir.txt" \
  "$GIT_REPO/outputs/logs/${RUN_NAME}_latest_log_dir.txt"
copy_if_exists \
  "$SOURCE_REPO/outputs/logs/${RUN_NAME}_latest_log_path.txt" \
  "$GIT_REPO/outputs/logs/${RUN_NAME}_latest_log_path.txt"
copy_if_exists \
  "$SOURCE_REPO/outputs/logs/${RUN_NAME}_success.txt" \
  "$GIT_REPO/outputs/logs/${RUN_NAME}_success.txt"
copy_if_exists \
  "$SOURCE_REPO/outputs/logs/${RUN_NAME}_failed.txt" \
  "$GIT_REPO/outputs/logs/${RUN_NAME}_failed.txt"
copy_if_exists \
  "$SOURCE_REPO/outputs/logs/${RUN_NAME}_running.txt" \
  "$GIT_REPO/outputs/logs/${RUN_NAME}_running.txt"

LATEST_LOG_DIR=""
LATEST_LOG_PATH=""

if [[ -f "$SOURCE_REPO/outputs/logs/${RUN_NAME}_latest_log_dir.txt" ]]; then
  LATEST_LOG_DIR="$(<"$SOURCE_REPO/outputs/logs/${RUN_NAME}_latest_log_dir.txt")"
fi
if [[ -f "$SOURCE_REPO/outputs/logs/${RUN_NAME}_latest_log_path.txt" ]]; then
  LATEST_LOG_PATH="$(<"$SOURCE_REPO/outputs/logs/${RUN_NAME}_latest_log_path.txt")"
fi

if [[ -n "$LATEST_LOG_DIR" ]]; then
  latest_dir_name="$(basename "$LATEST_LOG_DIR")"
  copy_dir_if_exists \
    "$SOURCE_REPO/$LATEST_LOG_DIR" \
    "$GIT_REPO/outputs/logs/$latest_dir_name"
fi

if [[ -n "$LATEST_LOG_PATH" ]]; then
  latest_log_name="$(basename "$LATEST_LOG_PATH")"
  copy_if_exists \
    "$SOURCE_REPO/$LATEST_LOG_PATH" \
    "$GIT_REPO/outputs/logs/$latest_log_name"
fi

cd "$GIT_REPO"

git add outputs/logs || true
if ! git diff --cached --quiet; then
  echo "[$(date '+%F %T')] sync_run_to_git: committing status snapshot"
  git commit -m "chore: sync ${RUN_NAME} status" || true
  echo "[$(date '+%F %T')] sync_run_to_git: pushing to origin/$branch"
  if ! git push origin "$branch"; then
    echo "[$(date '+%F %T')] sync_run_to_git: push failed" >&2
    exit 0
  fi
else
  echo "[$(date '+%F %T')] sync_run_to_git: no log changes to sync"
fi
