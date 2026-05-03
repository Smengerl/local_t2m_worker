#!/usr/bin/env bash
# =============================================================================
# create_examples_zimage.sh — Enqueue showcase jobs for all Z-Image-based configs.
#
# Usage (standalone):
#   ./examples/create_examples_zimage.sh
#   BASE_URL=http://localhost:9000 ./examples/create_examples_zimage.sh
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
    echo "  Z-Image showcase"
    echo "  Output : ${OUT_DIR}/<config-name>.png"
    echo "============================================================"
    echo
fi

echo "── Z-Image Turbo ───────────────────────────────────────────"

# Z-Image Turbo (8-step GGUF, photorealistic)
enqueue \
    "configs/zimage_turbo.json" \
    "a photorealistic portrait of a woman in soft evening light, warm golden background, sharp focus" \
    "zimage_turbo"

# Z-Image Turbo + Classic Painting Z LoRA (Old Masters oil-painting style)
enqueue \
    "configs/zimage_turbo_classic_painting_lora.json" \
    "class1cpa1nt, a 19th-century oil painting by Jan van Eyck, ethereal elegance, dark wavy updo, ruffled white lace, blue accents, pearl detail, dark textured backdrop, poised and wise." \
    "zimage_turbo_classic_painting_lora"

# Z-Image Turbo + 1950s American Dream LoRA (vintage Americana painting style)
enqueue \
    "configs/zimage_turbo_1950s_american_dream_lora.json" \
    "a cheerful american family at a drive-in diner, nuclear explosion mushroom cloud on the horizon seen through the window" \
    "zimage_turbo_1950s_american_dream_lora"

echo
