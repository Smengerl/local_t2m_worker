#!/usr/bin/env bash
# helpers/worker_status.sh — Worker process helpers
#
# Functions provided:
#   batch_instance_running — true if batch.server OR batch.worker holds batch.lock
#   start_worker_bg        — starts batch.worker in background, waits until lock is held
#
# USAGE: Source this file after ROOT_DIR and env.sh are set up.

# ── Locate project root (two levels up from scripts/helpers/) ─────────────────
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Ensure platform helpers are available (safe to source multiple times)
source "$ROOT_DIR/scripts/helpers/platform.sh"

WORKER_LOG="${WORKER_LOG:-$ROOT_DIR/batch/worker.log}"

# ── batch_instance_running ────────────────────────────────────────────────────
# Returns true if batch.server OR batch.worker is currently running.
#
# Uses the exclusive batch.lock file as the single source of truth:
#   - Both batch.server and batch.worker acquire this lock at startup.
#   - The OS releases it automatically when the process exits (even on crash).
#   - No stale-file cleanup, no PID tracking, no platform-specific process
#     listing required.
#
# Replaces the former worker_running() + server_process_running() pair.
batch_instance_running() {
  # Spawn a tiny Python snippet that tries to acquire batch.lock with timeout=0.
  # Timeout → lock is held → a process is running → exit 0 (shell true).
  # Success → lock is free → nothing running        → exit 1 (shell false).
  ROOT_DIR="$ROOT_DIR" "$PYTHON" - <<'PYEOF' 2>/dev/null
import os, sys
from pathlib import Path
from filelock import FileLock, Timeout
lock_path = Path(os.environ["ROOT_DIR"]) / "batch.lock"
lk = FileLock(str(lock_path), timeout=0)
try:
    lk.acquire()
    lk.release()
    sys.exit(1)   # lock free  → nothing running → shell false
except Timeout:
    sys.exit(0)   # lock held  → instance alive  → shell true
PYEOF
}

# ── start_worker_bg ───────────────────────────────────────────────────────────
# Starts batch.worker in background and waits up to 10 s for it to acquire
# batch.lock (= proof that it started successfully).
# Requires: PYTHON, WORKER_LOG to be set.
start_worker_bg() {
  nohup "$PYTHON" -m batch.worker >> "$WORKER_LOG" 2>&1 &
  local i
  for i in {1..40}; do
    sleep 0.25
    if batch_instance_running; then break; fi
  done
  if batch_instance_running; then
    echo "✅ Worker started (log: $WORKER_LOG)."
  else
    echo "❌ Worker could not be started. Check $WORKER_LOG for errors." >&2
    # Print last 20 lines of log to stderr for quick debugging (e.g. missing venv, import error)
    tail -20 "$WORKER_LOG" >&2
    exit 1
  fi
}
