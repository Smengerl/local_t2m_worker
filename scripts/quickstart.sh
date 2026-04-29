#!/usr/bin/env bash
# scripts/quickstart.sh — Cross-platform first-time setup for local_t2m_worker
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=helpers/env.sh
source "$ROOT_DIR/scripts/helpers/env.sh"
# (platform.sh is sourced transitively by env.sh)

# Step 1: Create and activate venv (auto-create if missing)
echo "[1/3] Setting up Python virtual environment..."
activate_venv --auto-create
resolve_venv_python

# Step 2: Optionally store HuggingFace token
read -rp "[2/3] Enter your HuggingFace token (or leave blank to skip): " HF_TOKEN
if [[ -n "$HF_TOKEN" ]]; then
  echo "$HF_TOKEN" > "$ROOT_DIR/.hf_token"
  echo "Saved HuggingFace token to .hf_token"
fi

# Step 3: Print usage instructions
printf "\n[3/3] Setup complete!\n"
echo "To generate an image, run:"
if is_windows; then
  echo "    .\\scripts\\run.sh \"a misty forest at dawn\""
else
  echo "    ./scripts/run.sh \"a misty forest at dawn\""
fi
