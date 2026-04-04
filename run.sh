#!/usr/bin/env zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# ── Usage hint ────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $0 [OPTIONS] "your prompt here"

Options:
  -c, --config FILE          JSON config file (default: configs/sd15_default.json)
  -n, --negative-prompt TEXT Negative prompt
  -o, --output FILE          Output PNG path (default: outputs/<timestamp>.png)
      --model-id REPO_ID     Override model ID from config
      --adapter-id REPO_ID   Override adapter ID from config
      --lora-id REPO_ID      Override LoRA weights from config
      --lora-scale FLOAT     Override LoRA scale from config
      --steps N              Override inference steps from config
      --guidance-scale FLOAT Override guidance scale from config
  -h, --help                 Show this help

Examples:
  $0 "a sunset over the ocean"
  $0 -c configs/sdxl_graffiti_lora.json -o outputs/dragon.png "graffiti mural of a dragon"
  $0 -c configs/sd15_default.json --steps 50 --guidance-scale 8.0 "a cat"
EOF
  exit 0
}

# Show help if no arguments provided
[[ $# -eq 0 ]] && usage

# Pass --help through to Python (which prints its own help) or handle -h here
for arg in "$@"; do
  [[ "$arg" == "-h" || "$arg" == "--help" ]] && usage
done
# ─────────────────────────────────────────────────────────────────────────────

# ── Activate virtual environment ─────────────────────────────────────────────
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Virtual environment not found. Creating it now..."
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  echo "Installing dependencies..."
  pip install --upgrade pip -q
  pip install -r "$SCRIPT_DIR/requirements.txt"
else
  source "$VENV/bin/activate"
fi

# ── Optional: HF token login (needed for gated models) ───────────────────────
# Priority order:
#   1. .hf_token file in project root  (one line: your token)
#   2. HF_TOKEN environment variable
#   3. Interactive huggingface-cli login (if not already logged in)
TOKEN_FILE="$SCRIPT_DIR/.hf_token"
HF_CLI="$VENV/bin/huggingface-cli"

if [[ ! -f "$HF_CLI" ]]; then
  echo "⚠️  huggingface-cli not found – installing huggingface_hub..."
  pip install huggingface_hub -q
fi

if [[ -f "$TOKEN_FILE" ]]; then
  echo "🔑 Using HF token from .hf_token file..."
  HF_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
  "$HF_CLI" login --token "$HF_TOKEN" 2>/dev/null \
    && echo "✅ Logged in to Hugging Face." \
    || echo "⚠️  Token login failed – check .hf_token."
elif [[ -n "$HF_TOKEN" ]]; then
  echo "🔑 Using HF_TOKEN from environment..."
  "$HF_CLI" login --token "$HF_TOKEN" 2>/dev/null \
    && echo "✅ Logged in to Hugging Face." \
    || echo "⚠️  Token login failed – check HF_TOKEN."
else
  if ! "$HF_CLI" whoami &>/dev/null; then
    echo "ℹ️  No HF token found. Launching interactive login..."
    echo "    (skip with Ctrl+C if you only use public models)"
    "$HF_CLI" login || true
  else
    echo "✅ Already logged in to Hugging Face ($("$HF_CLI" whoami 2>/dev/null | head -1))."
  fi
fi
# ─────────────────────────────────────────────────────────────────────────────

# ── Run the generator ─────────────────────────────────────────────────────────
echo "Python: $(which python)"

# If no --config flag was given by the user, inject the default config
has_config=0
for arg in "$@"; do
  [[ "$arg" == "--config" || "$arg" == "-c" ]] && has_config=1 && break
done

if [[ $has_config -eq 0 ]]; then
  DEFAULT_CONFIG="$SCRIPT_DIR/configs/sd15_default.json"
  echo "ℹ️  No --config specified, using default: $DEFAULT_CONFIG"
  python "$SCRIPT_DIR/generate.py" --config "$DEFAULT_CONFIG" "$@"
else
  python "$SCRIPT_DIR/generate.py" "$@"
fi