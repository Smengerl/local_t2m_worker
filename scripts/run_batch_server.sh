#!/usr/bin/env bash
# run_batch_server.sh — Start the worker + web server together.
#
# Usage:
#   ./run_batch_server.sh              # localhost:8000
#   ./run_batch_server.sh --offline    # skip HuggingFace update checks
#   PORT=9000 ./run_batch_server.sh    # custom port
#
# After `pip install -e .` you can also run without this wrapper:
#   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 HF_TOKEN=$(cat .hf_token) t2m-server --port 8000

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8000}"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"

for arg in "$@"; do
  [[ "$arg" == "--offline" ]] && apply_offline_mode
done

apply_pytorch_mps_env
load_hf_token
activate_venv

echo "▶ Starting server + in-process worker on http://localhost:${PORT}"
echo "   Press Ctrl-C to stop."
echo ""

"$VENV_BIN/t2m-server" --port "$PORT"
