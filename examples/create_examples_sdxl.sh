#!/usr/bin/env bash
# =============================================================================
# create_examples_sdxl.sh — Enqueue showcase jobs for all SDXL-based configs.
#
# Usage (standalone):
#   ./examples/create_examples_sdxl.sh
#   BASE_URL=http://localhost:9000 ./examples/create_examples_sdxl.sh
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
    echo "  SDXL showcase"
    echo "  Output : ${OUT_DIR}/<config-name>.png"
    echo "============================================================"
    echo
fi

echo "── SDXL ────────────────────────────────────────────────────"

# SDXL Turbo (1–4 steps, adversarial distillation)
enqueue \
    "configs/sdxl_turbo.json" \
    "a photorealistic tabby cat sitting on a windowsill in warm afternoon light" \
    "sdxl_turbo"

# SDXL Hyper-SD LoRA (8-step distillation)
enqueue \
    "configs/sdxl_hypersd_lora.json" \
    "a photorealistic portrait of a woman with green eyes, professional studio lighting, bokeh background" \
    "sdxl_hypersd_lora"

# SDXL Analog Redmond LoRA (analog film photography look)
enqueue \
    "configs/sdxl_analog_redmond_lora.json" \
    "AnalogRedmAF portrait of a young man at golden hour, film grain, warm tones, 35mm photograph" \
    "sdxl_analog_redmond_lora"

# SDXL B&W Manga LoRA (bold ink manga style)
enqueue \
    "configs/sdxl_bandw_manga_lora.json" \
    "a boy in a sailor school uniform standing on a rooftop at sunset, manga illustration, bold ink lines" \
    "sdxl_bandw_manga_lora"

# SDXL Graffiti LoRA (spray-paint mural)
enqueue \
    "configs/sdxl_graffiti_lora.json" \
    "graarg graffiti mural of a roaring lion on a brick wall, vibrant spray-paint colours" \
    "sdxl_graffiti_lora"

# SDXL IKEA LoRA (minimalist flat assembly-style illustration)
enqueue \
    "configs/sdxl_ikea_lora.json" \
    "a bicycle, flat minimalist assembly diagram illustration, black lines on white background" \
    "sdxl_ikea_lora"

# SDXL LittleTinies LoRA (soft childlike hand-drawn)
enqueue \
    "configs/sdxl_littletinies_lora.json" \
    "a tiny witch child riding a broomstick over a moonlit village, soft pastel colours, hand-drawn style" \
    "sdxl_littletinies_lora"

# SDXL Papercut LoRA (scherenschnitt paper-cut art)
enqueue \
    "configs/sdxl_papercut_lora.json" \
    "papercut forest with deer and mountains, intricate layered paper silhouettes, white background" \
    "sdxl_papercut_lora"

# SDXL Pokémon Trainer Sprite LoRA
enqueue \
    "configs/sdxl_pokemon_sprite_lora.json" \
    "a Pokémon trainer sprite in pixel art style, gen1 palette, facing forward" \
    "sdxl_pokemon_sprite_lora"

# SDXL Storyboard Sketch LoRA (grayscale film pre-production)
enqueue \
    "configs/sdxl_storyboard_sketch_lora.json" \
    "storyboard sketch of a hero running through a collapsing building, dynamic camera angle, grayscale pencil" \
    "sdxl_storyboard_sketch_lora"

# SDXL Watercolor LoRA (wet-on-wet watercolour painting)
enqueue \
    "configs/sdxl_watercolor_lora.json" \
    "a mountain landscape at sunset with a winding river, loose watercolour painting, soft bleeding edges" \
    "sdxl_watercolor_lora"

echo

# ---------------------------------------------------------------------------
