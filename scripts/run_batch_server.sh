#!/usr/bin/env bash
# run_batch_server.sh – Start the worker + web server together.
#
# Usage:
#   ./run_batch_server.sh              # localhost:8000
#   ./run_batch_server.sh --offline    # skip HuggingFace update checks
#   PORT=9000 ./run_batch_server.sh    # custom port

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8000}"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"

# ── Parse flags ───────────────────────────────────────────────────────────────
OFFLINE=false
for arg in "$@"; do
  [[ "$arg" == "--offline" ]] && OFFLINE=true
done

# Apply offline mode: set env var so huggingface_hub skips all network calls
if [[ "$OFFLINE" == true ]]; then
  apply_offline_mode
fi

# Apply MPS memory settings on macOS
apply_pytorch_mps_env

# Load HuggingFace token from .hf_token (required for gated models like FLUX)
load_hf_token

# Activate virtual environment and resolve python
activate_venv
resolve_venv_python

echo "▶ Starting server + in-process worker on http://localhost:${PORT}"
echo "   (Worker runs inside the server process — no separate PID to manage.)"
echo "   Press Ctrl-C to stop."
echo ""

"$PYTHON" -m batch.server --port "$PORT"
