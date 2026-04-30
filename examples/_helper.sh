#!/usr/bin/env bash
# =============================================================================
# _helper.sh — Shared helpers for the create_examples_* scripts.
#
# Source this file — do NOT execute it directly.
#
# Provides:
#   enqueue  <config>  <prompt>  <output_name>
#       Posts one job to the batch REST API and prints a status line.
#
#   check_server
#       Verifies the batch server is reachable; exits with a message if not.
#
# Environment:
#   BASE_URL    Server base URL (default: http://localhost:8000)
#   QUEUE_MODE  Set to true (default) to enqueue jobs via batch queue,
#               false for direct synchronous generation.
#
# Populates:
#   SCRIPT_DIR  Absolute path to the examples/ directory
#   OUT_DIR     Same as SCRIPT_DIR — output images land here
# =============================================================================

# Set QUEUE_MODE=true to enqueue jobs (default), false for direct synchronous generation
QUEUE_MODE="${QUEUE_MODE:-true}"


# Resolve the examples/ directory from whichever script sourced this file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SCRIPT_DIR"

# Load shared helpers — provides batch_instance_running() and PYTHON variable
# (env.sh → activate_venv + resolve_venv_python; worker_status.sh → batch_instance_running)
ROOT_DIR="$REPO_ROOT"
source "$REPO_ROOT/scripts/helpers/env.sh"
source "$REPO_ROOT/scripts/helpers/worker_status.sh"
activate_venv
resolve_venv_python

# Source health_check.sh to get run_checks() for wait_for_queue display.
# The BASH_SOURCE guard inside health_check.sh prevents the loop from starting.
source "$REPO_ROOT/scripts/health_check.sh"

# ---------------------------------------------------------------------------
# enqueue <config> <prompt> <output_name>
#
#   config       Path to the JSON config file, e.g. "configs/flux_schnell.json"
#   prompt       Generation prompt (will be JSON-escaped automatically)
#   output_name  Basename without extension; image is saved as OUT_DIR/<name>.png
# ---------------------------------------------------------------------------
enqueue() {
    local config="$1"
    local prompt="$2"
    local output_name="$3"

    local output_path="${OUT_DIR}/${output_name}.png"

    # Always use absolute paths for script and config
    local abs_config="${REPO_ROOT}/$config"
    local abs_run_sh="${REPO_ROOT}/scripts/run.sh"

    echo "[DEBUG] enqueue() called with:"
    echo "        config      = $config"
    echo "        prompt      = $prompt"
    echo "        output_path = $output_path"

    set -x
    # Always run from repo root so Python can import batch.*
    if [[ "$QUEUE_MODE" == true ]]; then
        if (cd "$REPO_ROOT" && "$abs_run_sh" --queue -c "$abs_config" -o "$output_path" "$prompt"); then
            set +x
            printf "  ✅  %-46s  →  enqueued\n" "$output_name"
        else
            set +x
            printf "  ❌  %-46s  →  enqueue failed\n" "$output_name"
        fi
    else
        if (cd "$REPO_ROOT" && "$abs_run_sh" -c "$abs_config" -o "$output_path" "$prompt"); then
            set +x
            printf "  ✅  %-46s  →  generated\n" "$output_name"
        else
            set +x
            printf "  ❌  %-46s  →  generation failed\n" "$output_name"
        fi
    fi
}

# ---------------------------------------------------------------------------
# wait_for_queue
#
#   Shows run_checks() display every 5 s until batch.lock is released.
#   Only active when QUEUE_MODE=true; no-op in direct mode.
# ---------------------------------------------------------------------------
wait_for_queue() {
    [[ "$QUEUE_MODE" != true ]] && return 0

    # If no worker/server is running, skip silently.
    if ! batch_instance_running; then
        return 0
    fi

    trap 'printf "\nStopped watching.\n"; return 0' INT TERM

    while batch_instance_running; do
        run_checks
        sleep 5
    done

    # Final display after lock is released — shows end state.
    run_checks
    printf "\n  ✅  Worker finished — batch.lock released.\n"

    trap - INT TERM
}
