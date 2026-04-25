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

# ── Venv paths (Unix: bin/, Windows/Git-Bash: Scripts/) ──────────────────────
VENV="$ROOT_DIR/.venv"
if [[ -d "$VENV/Scripts" ]]; then
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
      if [[ -d "$VENV/Scripts" ]]; then VENV_BIN="$VENV/Scripts"; else VENV_BIN="$VENV/bin"; fi
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

# ── resolve_venv_python ───────────────────────────────────────────────────────
# Sets PYTHON to the venv's python executable after activate_venv was called.
resolve_venv_python() {
  PYTHON="$VENV_BIN/python"
  [[ -x "$PYTHON" ]] || PYTHON="$VENV_BIN/python3"
}
