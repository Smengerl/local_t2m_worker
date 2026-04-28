#!/usr/bin/env bash
# =============================================================================
# create_examples_sd3.sh — Enqueue showcase jobs for all SD3-based configs.
#
# Usage (standalone):
#   ./examples/create_examples_sd3.sh
#   BASE_URL=http://localhost:9000 ./examples/create_examples_sd3.sh
#
# Can also be sourced by create_examples_all.sh (server check is skipped then).
# =============================================================================

set -euo pipefail

# shellcheck source=_helper.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_helper.sh"

# ---------------------------------------------------------------------------
# Standalone guard
# ---------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo
    echo "============================================================"
    echo "  Stable Diffusion 3 showcase"
    echo "  Output : ${OUT_DIR}/<config-name>.png"
    echo "============================================================"
    echo
fi

echo "── Stable Diffusion 3 ──────────────────────────────────────"

# SD3 Medium — high-quality generation with accurate text rendering
enqueue \
    "configs/sd3_medium.json" \
    "a vibrant street market in Marrakech at golden hour, detailed textures, rich colours, cinematic composition" \
    "sd3_medium"

echo

# ---------------------------------------------------------------------------
