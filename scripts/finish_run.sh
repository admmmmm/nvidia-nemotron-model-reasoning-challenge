#!/usr/bin/env bash
set -euo pipefail

STATUS="${1:-unknown}"
RUN_NAME="${2:-run}"
LOG_FILE="${3:-}"
OUTPUT_DIR="${4:-}"
RUN_LOG_DIR="${5:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env.server ]]; then
  # shellcheck disable=SC1091
  source .env.server
fi

timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
marker_file="outputs/logs/${RUN_NAME}_$(echo "$STATUS" | tr '[:upper:]' '[:lower:]').txt"
mkdir -p outputs/logs
{
  echo "run_name=$RUN_NAME"
  echo "status=$STATUS"
  echo "timestamp=$timestamp"
  echo "log_file=$LOG_FILE"
  echo "output_dir=$OUTPUT_DIR"
  echo "run_log_dir=$RUN_LOG_DIR"
} > "$marker_file"

if [[ -n "$RUN_LOG_DIR" ]]; then
  mkdir -p "$RUN_LOG_DIR"
  run_marker_file="${RUN_LOG_DIR}/status_$(echo "$STATUS" | tr '[:upper:]' '[:lower:]').txt"
  {
    echo "run_name=$RUN_NAME"
    echo "status=$STATUS"
    echo "timestamp=$timestamp"
    echo "log_file=$LOG_FILE"
    echo "output_dir=$OUTPUT_DIR"
    echo "run_log_dir=$RUN_LOG_DIR"
  } > "$run_marker_file"
fi

if [[ "${AUTO_GIT_PUSH:-0}" == "1" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  branch="${GIT_BRANCH:-main}"
  commit_message="${GIT_COMMIT_MESSAGE:-chore: update training artifacts for $RUN_NAME ($STATUS)}"
  git add PROJECT_PLAN.md outputs/logs scripts .env.server.example src || true
  if ! git diff --cached --quiet; then
    git commit -m "$commit_message" || true
    git push origin "$branch" || true
  fi
fi

if [[ -n "${WEBHOOK_URL:-}" ]]; then
  curl -sS -X POST "$WEBHOOK_URL"     --data-urlencode "title=$RUN_NAME"     --data-urlencode "desp=status: $STATUS
log: $LOG_FILE
output: $OUTPUT_DIR"     >/dev/null || true
fi

if [[ -n "${BARK_URL:-}" ]]; then
  curl -sS -G "$BARK_URL/$RUN_NAME/$STATUS"     --data-urlencode "body=log: ${LOG_FILE:-none} output: ${OUTPUT_DIR:-none}"     >/dev/null || true
fi

if [[ "${AUTO_SHUTDOWN:-0}" == "1" ]]; then
  if [[ -n "${STOP_COMMAND:-}" ]]; then
    bash -lc "$STOP_COMMAND" || true
  else
    sudo shutdown -h now || shutdown -h now || true
  fi
fi
