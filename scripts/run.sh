#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"

# ── Usage hint ────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $0 [OPTIONS] "your prompt here"

Options:
  -c, --config FILE          JSON config file (default: configs/sd15_default.json)
  -n, --negative-prompt TEXT Negative prompt
  -o, --output FILE          Output PNG path (default: outputs/<timestamp>.png)
      --model-id REPO_ID     Override model ID from config
      --lora-id REPO_ID      Override LoRA weights from config
      --lora-scale FLOAT     Override LoRA scale from config
      --steps N              Override inference steps from config
      --guidance-scale FLOAT Override guidance scale from config
      --queue                Add job to batch queue instead of generating immediately
      --offline              Skip HuggingFace update checks (use local cache only,
                             no network calls). Faster startup when models are
                             already downloaded. Fails if a model is not cached.
  -h, --help                 Show this help

Examples:
  $0 "a sunset over the ocean"
  $0 -c configs/sdxl_graffiti_lora.json -o outputs/dragon.png "graffiti mural of a dragon"
  $0 -c configs/sd15_default.json --steps 50 --guidance-scale 8.0 "a cat"
  $0 --queue -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"
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
activate_venv --auto-create
resolve_venv_python

# ── Optional: HF token (needed for gated models like FLUX, SD3) ──────────────
# diffusers reads HF_TOKEN from the environment automatically — no CLI login needed.
# Priority: .hf_token file > HF_TOKEN env var > already cached credentials
TOKEN_FILE="$ROOT_DIR/.hf_token"

if [[ -f "$TOKEN_FILE" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
  echo "🔑 HF token loaded from .hf_token."
elif [[ -n "${HF_TOKEN:-}" ]]; then
  echo "🔑 HF token found in environment."
else
  echo "ℹ️  No HF token found — gated models (FLUX, SD3) will not be accessible."
fi
# ─────────────────────────────────────────────────────────────────────────────

# ── Run the generator ─────────────────────────────────────────────────────────
echo "Python: $PYTHON"

# ── Strip --queue / --offline flags and collect remaining args ────────────────
QUEUE=false
OFFLINE=false
PASSTHROUGH_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--queue" ]]; then
    QUEUE=true
  elif [[ "$arg" == "--offline" ]]; then
    OFFLINE=true
  else
    PASSTHROUGH_ARGS+=("$arg")
  fi
done

# Apply offline mode: set env var so huggingface_hub skips all network calls
if [[ "$OFFLINE" == true ]]; then
  export HF_HUB_OFFLINE=1
  echo "📴 Offline mode enabled — skipping HuggingFace update checks."
fi

# Allow MPS to use the full unified memory pool (including swap).
# Without this macOS enforces a hard cap and kills the process on OOM.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# If no --config flag was given by the user, inject the default config
has_config=0
for arg in "${PASSTHROUGH_ARGS[@]}"; do
  [[ "$arg" == "--config" || "$arg" == "-c" ]] && has_config=1 && break
done

if [[ "$QUEUE" == true ]]; then
  # ── Queue mode: ensure worker is running, then hand off ───────────────────
  WORKER_PID_FILE="$ROOT_DIR/batch/worker.pid"
  WORKER_LOG="$ROOT_DIR/batch/worker.log"

  _worker_running() {
    [[ -f "$WORKER_PID_FILE" ]] || return 1
    local pid
    pid=$(cat "$WORKER_PID_FILE")
    kill -0 "$pid" 2>/dev/null
  }

  if _worker_running; then
    echo "✅ Worker already running (pid $(cat "$WORKER_PID_FILE"))."
  else
    echo "🚀 Worker not running — starting it in the background..."
    nohup "$PYTHON" -m batch.worker >> "$WORKER_LOG" 2>&1 &
    STARTED_WORKER_PID=$!
    # Give it a moment to write its PID file before we proceed
    for i in {1..10}; do
      sleep 0.3
      _worker_running && break
    done
    if _worker_running; then
      echo "✅ Worker started (pid $(cat "$WORKER_PID_FILE"), log: $WORKER_LOG)."
    else
      echo "⚠️  Worker may not have started correctly — check $WORKER_LOG"
    fi
  fi

  echo "📋 Adding job to batch queue..."
  if [[ $has_config -eq 0 ]]; then
    DEFAULT_CONFIG="$ROOT_DIR/configs/sd15_default.json"
    "$PYTHON" -m batch.enqueue --config "$DEFAULT_CONFIG" "${PASSTHROUGH_ARGS[@]}"
  else
    "$PYTHON" -m batch.enqueue "${PASSTHROUGH_ARGS[@]}"
  fi
else
  # ── Direct mode: generate immediately ─────────────────────────────────────
  if [[ $has_config -eq 0 ]]; then
    DEFAULT_CONFIG="$ROOT_DIR/configs/sd15_default.json"
    echo "ℹ️  No --config specified, using default: $DEFAULT_CONFIG"
    "$PYTHON" "$ROOT_DIR/generate.py" --config "$DEFAULT_CONFIG" "${PASSTHROUGH_ARGS[@]}"
  else
    "$PYTHON" "$ROOT_DIR/generate.py" "${PASSTHROUGH_ARGS[@]}"
  fi
fi