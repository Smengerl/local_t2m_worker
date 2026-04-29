#!/usr/bin/env bash
# helpers/env.sh — Shared environment bootstrap for all project scripts.
#
# USAGE: Source this file AFTER setting ROOT_DIR in the calling script:
#   ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
#   source "$ROOT_DIR/scripts/helpers/env.sh"
#
# Exports after sourcing:
#   SYS_PYTHON   — system python3 or python command
#   SYS_PIP      — system pip3 or pip command
#   VENV         — path to the virtual environment root
#   VENV_BIN     — path to venv's bin/ (Unix) or Scripts/ (Windows/Git-Bash)
#
# Functions provided:
#   activate_venv [--auto-create]
#     Activates the venv. With --auto-create it creates and populates it
#     from requirements.txt if it does not yet exist.
#
#   resolve_venv_python
#     Sets PYTHON to the venv's python executable (python or python3).
#
#   load_hf_token
#     Loads the HuggingFace token from .hf_token into HF_TOKEN env var.
#
#   apply_offline_mode
#     Sets HF_HUB_OFFLINE=1 and prints a message. Call after parsing --offline.
#
#   apply_pytorch_mps_env
#     Sets MPS memory env vars on macOS (no-op on other platforms).

# ── Locate project root (two levels up from scripts/helpers/) ─────────────────
# The fallback is used when this file is sourced without ROOT_DIR being set.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# ── Platform helpers (sourced first so all functions below can use is_windows etc.) ──
source "$ROOT_DIR/scripts/helpers/platform.sh"

# ── Detect system Python (python3 preferred, fallback to python) ──────────────
if command -v python3 >/dev/null 2>&1; then
  SYS_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  SYS_PYTHON="python"
else
  echo "❌ Python not found. Please install Python 3." >&2
  return 1 2>/dev/null || exit 1
fi

# ── Detect pip (pip3 preferred, fallback to pip) ──────────────────────────────
if command -v pip3 >/dev/null 2>&1; then
  SYS_PIP="pip3"
elif command -v pip >/dev/null 2>&1; then
  SYS_PIP="pip"
else
  echo "❌ pip not found. Please install pip." >&2
  return 1 2>/dev/null || exit 1
fi

# ── Venv paths (Unix: bin/, Windows/Git-Bash: Scripts/) ─────────────────-----
VENV="$ROOT_DIR/.venv"
if is_windows; then
  VENV_BIN="$VENV/Scripts"
else
  VENV_BIN="$VENV/bin"
fi

# ── activate_venv [--auto-create] ────────────────────────────────────────────
activate_venv() {
  local auto_create=false
  [[ "${1:-}" == "--auto-create" ]] && auto_create=true

  if [[ ! -f "$VENV_BIN/activate" ]]; then
    if $auto_create; then
      echo "Virtual environment not found. Creating it now..."
      "$SYS_PYTHON" -m venv "$VENV"
      # Refresh VENV_BIN after creation (Windows may differ)
      if is_windows; then VENV_BIN="$VENV/Scripts"; else VENV_BIN="$VENV/bin"; fi
      # shellcheck source=/dev/null
      source "$VENV_BIN/activate"
      echo "Installing dependencies..."
      "$SYS_PIP" install --upgrade pip -q
      "$SYS_PIP" install -r "$ROOT_DIR/requirements.txt"
    else
      echo "❌ Virtual environment not found at $VENV" >&2
      echo "   Run setup first: $SYS_PYTHON -m venv .venv && pip install -r requirements.txt" >&2
      return 1 2>/dev/null || exit 1
    fi
  else
    # shellcheck source=/dev/null
    source "$VENV_BIN/activate"
  fi
}

# ── resolve_venv_python ─────────────────────────────────────────────────-----
# Sets PYTHON to the venv's python executable after activate_venv was called.
resolve_venv_python() {
  if is_windows; then
    PYTHON="$VENV_BIN/python.exe"
    if [[ ! -x "$PYTHON" && -x "$VENV_BIN/python3.exe" ]]; then
      PYTHON="$VENV_BIN/python3.exe"
    fi
  else
    PYTHON="$VENV_BIN/python"
    if [[ ! -x "$PYTHON" && -x "$VENV_BIN/python3" ]]; then
      PYTHON="$VENV_BIN/python3"
    fi
  fi
}

# ── load_hf_token ─────────────────────────────────────────────────────────────
# Reads .hf_token from ROOT_DIR into HF_TOKEN. No-op if already set in env.
load_hf_token() {
  local token_file="$ROOT_DIR/.hf_token"
  if [[ -f "$token_file" ]]; then
    export HF_TOKEN
    # Strip all whitespace AND carriage returns (\r) — the latter occur when the
    # file was created or edited on Windows (CRLF line endings).  tr -d '[:space:]'
    # removes spaces, tabs and newlines but does NOT remove bare \r on some
    # platforms, so we strip \r explicitly first.
    HF_TOKEN="$(tr -d '\r' < "$token_file" | tr -d '[:space:]')"
    echo "🔑 HF token loaded from .hf_token."
  elif [[ -n "${HF_TOKEN:-}" ]]; then
    echo "🔑 HF token found in environment."
  else
    echo "ℹ️  No HF token found — gated models (FLUX, SD3) will not be accessible."
  fi
}

# ── apply_offline_mode ────────────────────────────────────────────────────────
# Sets HF_HUB_OFFLINE=1 and prints a notice. Call after parsing --offline flag.
apply_offline_mode() {
  export HF_HUB_OFFLINE=1
  echo "📴 Offline mode enabled — skipping HuggingFace update checks."
}

# ── apply_pytorch_mps_env ─────────────────────────────────────────────────────
# Sets MPS memory environment variables on macOS. No-op on other platforms.
apply_pytorch_mps_env() {
  if is_macos; then
    # Allow MPS to use the full unified memory pool (including swap).
    # Without this macOS enforces a hard cap and kills the process on OOM.
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  fi
}
