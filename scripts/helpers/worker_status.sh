#!/usr/bin/env bash
# worker_status.sh — Helper to check if the local CLI worker is running
# Usage: source this file and call worker_running

WORKER_PID_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/../batch/worker.pid"

worker_running() {
  [[ -f "$WORKER_PID_FILE" ]] || return 1
  local pid
  pid=$(cat "$WORKER_PID_FILE")
  # Check if process exists
  kill -0 "$pid" 2>/dev/null || return 1
  # Check if it's really a batch.worker process
  ps -p "$pid" -o command= | grep -q "batch.worker" || return 1
  return 0
}
