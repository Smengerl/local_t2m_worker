#!/usr/bin/env zsh
# run_batch_server.sh – Start the worker + web server together.
#
# Usage:
#   ./run_batch_server.sh              # localhost:8000
#   PORT=9000 ./run_batch_server.sh    # custom port
#
# The worker runs in the background and is killed automatically
# when this script exits (Ctrl-C or normal exit).

set -euo pipefail

SCRIPT_DIR="${0:A:h}"          # absolute dir of this script
PORT="${PORT:-8000}"

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Resolve python executable (venv may only provide python3 on some systems)
PYTHON="$SCRIPT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$SCRIPT_DIR/.venv/bin/python3"

echo "▶ Starting worker…"
"$PYTHON" -m batch.worker &
WORKER_PID=$!

# Make sure the worker is killed when this script exits
_cleanup() {
  echo ""
  echo "⏹ Stopping worker (pid $WORKER_PID)…"
  kill "$WORKER_PID" 2>/dev/null || true
  wait "$WORKER_PID" 2>/dev/null || true
  echo "✅ Done."
}
trap _cleanup EXIT INT TERM

echo "▶ Starting web server on http://localhost:${PORT}"
echo "   Press Ctrl-C to stop both."
echo ""

"$PYTHON" -m batch.server --port "$PORT"
