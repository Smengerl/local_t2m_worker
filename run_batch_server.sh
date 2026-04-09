#!/usr/bin/env zsh
# run_batch_server.sh – Start the worker + web server together.
#
# Usage:
#   ./run_batch_server.sh              # localhost:8000
#   ./run_batch_server.sh --offline    # skip HuggingFace update checks
#   PORT=9000 ./run_batch_server.sh    # custom port
#
# Options:
#   --offline   Set HF_HUB_OFFLINE=1 so huggingface_hub skips all network
#               calls. Faster startup when all models are already downloaded.
#               Fails if a model is not in the local cache.
#
# The worker runs in the background and is killed automatically
# when this script exits (Ctrl-C or normal exit).

set -euo pipefail

SCRIPT_DIR="${0:A:h}"          # absolute dir of this script
PORT="${PORT:-8000}"

# ── Parse flags ──────────────────────────────────────────────────────────────
OFFLINE=false
for arg in "$@"; do
  [[ "$arg" == "--offline" ]] && OFFLINE=true
done

if [[ "$OFFLINE" == true ]]; then
  export HF_HUB_OFFLINE=1
  echo "📴 Offline mode enabled — skipping HuggingFace update checks."
fi

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
