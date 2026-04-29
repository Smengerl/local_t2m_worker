#!/usr/bin/env bash
# helpers/worker_status.sh — Worker process helpers
#
# Functions provided:
#   worker_running        — true if the batch.worker PID file exists and the process is alive
#   server_process_running — true if a batch.server process is running (cross-platform)
#   start_worker_bg       — starts batch.worker in background, waits for PID file
#
# USAGE: Source this file after ROOT_DIR and env.sh are set up.

# ── Locate project root (two levels up from scripts/helpers/) ─────────────────
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Ensure platform helpers are available (safe to source multiple times)
source "$ROOT_DIR/scripts/helpers/platform.sh"

WORKER_PID_FILE="${WORKER_PID_FILE:-$ROOT_DIR/batch/worker.pid}"
WORKER_LOG="${WORKER_LOG:-$ROOT_DIR/batch/worker.log}"

# ── worker_running ────────────────────────────────────────────────────────────
worker_running() {
  [[ -f "$WORKER_PID_FILE" ]] || return 1
  local pid
  pid=$(cat "$WORKER_PID_FILE")
  if is_windows; then
    # Windows (Git Bash): use tasklist
    tasklist /FI "PID eq $pid" 2>/dev/null | grep -q "$pid" || return 1
    return 0
  else
    # Unix: check process liveness and that it is really batch.worker
    kill -0 "$pid" 2>/dev/null || return 1
    ps -p "$pid" -o command= 2>/dev/null | grep -q "batch.worker" || return 1
    return 0
  fi
}

# ── server_process_running ────────────────────────────────────────────────────
# Returns true if a batch.server Python process is running.
server_process_running() {
  if is_windows; then
    # wmic is available in Git Bash on Windows
    wmic process where "commandline like '%batch.server%'" get processid 2>/dev/null \
      | grep -qE '^[0-9]+' && return 0 || return 1
  else
    pgrep -f "batch\.server" >/dev/null 2>&1
  fi
}

# ── start_worker_bg ───────────────────────────────────────────────────────────
# Starts batch.worker in background and waits up to 10 s for the PID file.
# Requires: PYTHON, WORKER_LOG, WORKER_PID_FILE to be set.
start_worker_bg() {
  nohup "$PYTHON" -m batch.worker >> "$WORKER_LOG" 2>&1 &
  local i
  for i in {1..40}; do
    sleep 0.25
    if worker_running; then break; fi
  done
  if worker_running; then
    local pid=""
    [[ -f "$WORKER_PID_FILE" ]] && pid=$(cat "$WORKER_PID_FILE")
    echo "✅ Worker started (pid ${pid:-?}, log: $WORKER_LOG)."
  else
    # The worker failed to start within the timeout. Common reason: another
    # process already holds the exclusive lock (see batch/instance_lock.py).
    # In that case the job was still enqueued successfully and the existing
    # process (batch.server or batch.worker) will process the queue. Treat
    # this as a non-fatal condition to avoid reporting enqueue as failed.
    if server_process_running >/dev/null 2>&1 || pgrep -f "batch\.worker" >/dev/null 2>&1; then
      echo "⚠  Worker start skipped — another batch.server or batch.worker process is already running. " \
           "Assuming existing instance will process the queue (log: $WORKER_LOG)."
      return 0
    fi

    echo "❌ Worker could not be started. Check $WORKER_LOG for errors." >&2
    exit 1
  fi
}
