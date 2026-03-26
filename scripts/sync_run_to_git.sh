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

copy_if_exists \
  "$SOURCE_REPO/PROJECT_PLAN.md" \
  "$GIT_REPO/PROJECT_PLAN.md"
copy_if_exists \
  "$SOURCE_REPO/README.md" \
  "$GIT_REPO/README.md"
copy_if_exists \
  "$SOURCE_REPO/NEMOTRON_4BIT_MAMBA_SHAPE_MISMATCH_DEBUG.md" \
  "$GIT_REPO/NEMOTRON_4BIT_MAMBA_SHAPE_MISMATCH_DEBUG.md"
copy_dir_if_exists \
  "$SOURCE_REPO/scripts" \
  "$GIT_REPO/scripts"
copy_dir_if_exists \
  "$SOURCE_REPO/src" \
  "$GIT_REPO/src"

cd "$GIT_REPO"

branch="$(git branch --show-current 2>/dev/null || true)"
if [[ -z "$branch" ]]; then
  branch="master"
fi

git add outputs/logs PROJECT_PLAN.md README.md NEMOTRON_4BIT_MAMBA_SHAPE_MISMATCH_DEBUG.md scripts src .env.server.example || true
if ! git diff --cached --quiet; then
  git commit -m "chore: sync ${RUN_NAME} status" >/dev/null 2>&1 || true
  git push origin "$branch" >/dev/null 2>&1 || true
fi
