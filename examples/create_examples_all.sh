#!/usr/bin/env bash
# =============================================================================
# create_examples_all.sh — Orchestrator: enqueues one showcase job per non-NSFW
#                        config across all supported pipeline types.
#
# This script sources each per-pipeline script in sequence after a single
# server-reachability check.  You can also run any pipeline script on its own:
#
#   ./examples/create_examples_flux.sh
#   ./examples/create_examples_sd.sh
#   ./examples/create_examples_sd3.sh
#   ./examples/create_examples_sdxl.sh
#   ./examples/create_examples_zimage.sh
#
# Usage:
#   ./examples/create_examples_all.sh            # default http://localhost:8000
#   BASE_URL=http://localhost:9000 ./examples/create_examples_all.sh
#
# Requirements:
#   - The batch server must be running  (./scripts/run_batch_server.sh)
#   - curl must be available
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load shared helper (enqueue + check_server + BASE_URL + OUT_DIR)
# shellcheck source=_helper.sh
source "${SCRIPT_DIR}/_helper.sh"

echo
echo "============================================================"
echo "  Full showcase — enqueueing one job per config"
echo "  Output : ${OUT_DIR}/<config-name>.png"
echo "============================================================"
echo


# ===========================================================================
# ── FLUX models ─────────────────────────────────────────────────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# Source each pipeline script.  The BASH_SOURCE guard inside each script
# suppresses their standalone header/footer when they are sourced here.
# ---------------------------------------------------------------------------

# shellcheck source=create_examples_flux.sh
source "${SCRIPT_DIR}/create_examples_flux.sh"
# shellcheck source=create_examples_sd.sh
source "${SCRIPT_DIR}/create_examples_sd.sh"
# shellcheck source=create_examples_sd3.sh
source "${SCRIPT_DIR}/create_examples_sd3.sh"
# shellcheck source=create_examples_sdxl.sh
source "${SCRIPT_DIR}/create_examples_sdxl.sh"
# shellcheck source=create_examples_zimage.sh
source "${SCRIPT_DIR}/create_examples_zimage.sh"

