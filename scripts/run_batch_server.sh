#!/usr/bin/env bash
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

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8000}"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"

# ── Parse flags ──────────────────────────────────────────────────────────────
OFFLINE=false
for arg in "$@"; do
  [[ "$arg" == "--offline" ]] && OFFLINE=true
done

if [[ "$OFFLINE" == true ]]; then
  export HF_HUB_OFFLINE=1
  echo "📴 Offline mode enabled — skipping HuggingFace update checks."
fi

# Allow MPS to use the full unified memory pool (including swap).
# Without this macOS enforces a hard cap and kills the process on OOM.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# Set a limit for MPS memory allocations (e.g., 80% of total RAM).
# This allows PyTorch to raise a catchable Python RuntimeError instead of 
# letting the macOS OOM Killer terminate the process with "Killed: 9".
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8

# Activate virtual environment and resolve python
activate_venv
resolve_venv_python

echo "▶ Starting server + in-process worker on http://localhost:${PORT}"
echo "   (Worker runs inside the server process — no separate PID to manage.)"
echo "   Press Ctrl-C to stop."
echo ""

"$PYTHON" -m batch.server --port "$PORT"
