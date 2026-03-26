#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${1:-/root/nemotron}"
GIT_REPO="${2:-/root/nemotron_git}"
RUN_NAME="${3:-nemotron_lora_v0}"
INTERVAL_SECONDS="${4:-60}"

while true; do
  bash "$SOURCE_REPO/scripts/sync_run_to_git.sh" "$SOURCE_REPO" "$GIT_REPO" "$RUN_NAME" || true
  sleep "$INTERVAL_SECONDS"
done
