#!/usr/bin/env bash
# =============================================================================
# create_examples_flux.sh — Enqueue showcase jobs for all FLUX-based configs.
#
# Usage (standalone):
#   ./examples/create_examples_flux.sh
#   BASE_URL=http://localhost:9000 ./examples/create_examples_flux.sh
#
# Can also be sourced by create_examples_all.sh (server check is skipped then).
# =============================================================================

set -euo pipefail

# shellcheck source=_helper.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_helper.sh"

# ---------------------------------------------------------------------------
# Standalone guard — check server only when executed directly, not when sourced
# ---------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo
    echo "============================================================"
    echo "  FLUX showcase"
    echo "  Output : ${OUT_DIR}/<config-name>.png"
    echo "============================================================"
    echo
fi

echo "── FLUX ────────────────────────────────────────────────────"

# FLUX.1-schnell (4-step, fast general-purpose)
enqueue \
    "configs/flux_schnell.json" \
    "a misty forest at dawn, sunrays filtering through tall pine trees, volumetric light, photorealistic" \
    "flux_schnell"

# FLUX.1-dev (20-step, high-quality general-purpose)
enqueue \
    "configs/flux_dev.json" \
    "a majestic snow-capped mountain reflected in a crystal-clear alpine lake, golden hour, hyperrealistic photography" \
    "flux_dev"

# FLUX.2-klein 4B (compact FLUX.2, 4-step)
enqueue \
    "configs/flux2_klein_4b.json" \
    "a golden sunset over mountain peaks, warm light painting the clouds in shades of orange and purple, cinematic" \
    "flux2_klein_4b"

# FLUX.1-schnell + cute-comic LoRA
enqueue \
    "configs/flux_cute_comic_lora.json" \
    "a brave little fox on an adventure through an enchanted forest, in the style of TOK" \
    "flux_cute_comic_lora"

# FLUX.1-dev + cute-comic LoRA (higher quality than schnell variant)
enqueue \
    "configs/flux_dev_cute_comic_lora.json" \
    "a cheerful robot baking cookies in a cosy kitchen, in the style of TOK" \
    "flux_dev_cute_comic_lora"

# FLUX.1-schnell + miniature-people LoRA
enqueue \
    "configs/flux_miniature_people_lora.json" \
    "miniature people hiking on a sandwich used as a mountain trail, macro photography, shallow depth of field" \
    "flux_miniature_people_lora"

# FLUX.1-dev + miniature-people LoRA (higher quality variant)
enqueue \
    "configs/flux_dev_miniature_people_lora.json" \
    "miniature people camping next to a candle flame at night, macro photography, bokeh" \
    "flux_dev_miniature_people_lora"

echo

# ---------------------------------------------------------------------------
