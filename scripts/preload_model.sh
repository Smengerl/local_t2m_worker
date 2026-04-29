#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_NAME="$0"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"
# (platform.sh is sourced transitively by env.sh)

# ── Usage hint ────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $SCRIPT_NAME -c CONFIG [CONFIG …] [OPTIONS]

Pre-download all HuggingFace model weights required by one or more pipeline
configs. Already-cached blobs are skipped; partially-downloaded blobs are
resumed — nothing is re-downloaded from scratch.

Options:
  -c, --config FILE [FILE …]   One or more JSON config files (required)
  --cache-dir DIR              Custom cache directory for weights.
                               Default: value from config, then ~/.cache/huggingface
  --token-file FILE            Path to HF token file.
                               Default: .hf_token in project root
  --dry-run                    Print what would be downloaded without
                               downloading anything
  -h, --help                   Show this help and exit

Examples:
  $SCRIPT_NAME -c configs/flux_schnell.json
  $SCRIPT_NAME -c configs/flux_schnell.json configs/sdxl_graffiti_lora.json
  $SCRIPT_NAME -c configs/flux_dev_gguf.json --cache-dir models/
  $SCRIPT_NAME -c configs/flux_schnell.json --dry-run

GGUF configs:
  For configs with a gguf_file field the script downloads only what the
  pipeline actually needs:
    model_id repo     → only the specific .gguf file + metadata
    base_model_id repo → everything except transformer/ (~24 GB saved)
    lora_id repo      → only the single file named by weight_name

After a successful download, pass --offline to run.sh / run_batch_server.sh
to skip all HuggingFace network checks on startup:
  ./scripts/run_batch_server.sh --offline
EOF
  exit 0
}

# Show help if no arguments provided or -h / --help passed
[[ $# -eq 0 ]] && usage
for arg in "$@"; do
  [[ "$arg" == "-h" || "$arg" == "--help" ]] && usage
done

# ── Activate virtual environment ──────────────────────────────────────────────
activate_venv
resolve_venv_python
cd "$ROOT_DIR"
exec "$PYTHON" "$ROOT_DIR/scripts/preload_model.py" "$@"
