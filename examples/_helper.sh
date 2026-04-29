# =============================================================================
# Queue mode toggle
# Set QUEUE_MODE=true to enqueue jobs (default), false for direct synchronous generation
QUEUE_MODE=true

# Kill running worker (if any)
kill_worker() {
    # Use the same logic as worker_running to find the PID file
    local pid_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/../batch/worker.pid"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing running worker (pid $pid)..."
            kill "$pid"
        fi
    fi
}
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
#   BASE_URL   Server base URL (default: http://localhost:8000)
#
# Populates:
#   SCRIPT_DIR  Absolute path to the examples/ directory
#   OUT_DIR     Same as SCRIPT_DIR — output images land here
# =============================================================================


# Resolve the examples/ directory from whichever script sourced this file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SCRIPT_DIR"

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
