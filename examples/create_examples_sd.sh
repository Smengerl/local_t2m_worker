#!/usr/bin/env bash
# =============================================================================
# create_examples_sd.sh — Enqueue showcase jobs for all SD-based configs.
#
# Covers three groups that all share the "sd" pipeline backend:
#   • Stable Diffusion 1.5  — base checkpoints and LoRAs
#   • Comic Diffusion       — SD 1.5 fine-tunes with artist-style LoRAs
#   • Stable Diffusion 2.1  — base checkpoint + LoRA
#
# Usage (standalone):
#   ./examples/create_examples_sd.sh
#   BASE_URL=http://localhost:9000 ./examples/create_examples_sd.sh
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
    echo "  Stable Diffusion showcase  (SD 1.5 · Comic · SD 2.1)"
    echo "  Output : ${OUT_DIR}/<config-name>.png"
    echo "============================================================"
    echo
fi

# ===========================================================================
# Stable Diffusion 1.5
# ===========================================================================

echo "── Stable Diffusion 1.5 ────────────────────────────────────"

# SD 1.5 default (vanilla)
enqueue \
    "configs/sd15_default.json" \
    "a photorealistic portrait of a young woman with freckles, soft natural window light, shallow depth of field" \
    "sd15_default"

# DreamShaper 8 (versatile artistic fine-tune)
enqueue \
    "configs/sd15_dreamshaper8.json" \
    "fantasy portrait of an elf warrior in enchanted armour, dramatic rim lighting, detailed digital painting" \
    "sd15_dreamshaper8"

# Realistic Vision v6 (photorealism)
enqueue \
    "configs/sd15_realistic_vision_v6.json" \
    "RAW photo, portrait of a young woman with auburn hair, soft studio lighting, sharp focus, 85mm lens" \
    "sd15_realistic_vision_v6"

# Inkpunk Diffusion DreamBooth LoRA
enqueue \
    "configs/sd15_inkpunk_lora.json" \
    "nvinkpunk portrait of a samurai warrior, ink brush strokes, cyberpunk city in the background" \
    "sd15_inkpunk_lora"

# Pixel Art LoRA
enqueue \
    "configs/sd15_pixel_art_lora.json" \
    "pixel art dragon on a mountain peak, 16-bit retro game style, sunset colours" \
    "sd15_pixel_art_lora"

echo

# ===========================================================================
# Comic Diffusion — SD 1.5 fine-tunes with artist-style LoRAs
# ===========================================================================

echo "── Comic Diffusion artist styles ───────────────────────────"

# Andreas Rocha — rich colour fantasy landscapes
enqueue \
    "configs/sd15_comic_diffusion_andreasrocha.json" \
    "a castle on a cliff overlooking a misty valley at sunset, andreasrocha artstyle" \
    "sd15_comic_diffusion_andreasrocha"

# Charlie Bo — geometric superhero panels
enqueue \
    "configs/sd15_comic_diffusion_charliebo.json" \
    "a superhero leaping across rooftops in a neon city at night, charliebo artstyle" \
    "sd15_comic_diffusion_charliebo"

# Hollie Mengert — soft storybook illustration
enqueue \
    "configs/sd15_comic_diffusion_holliemengert.json" \
    "a fairy sitting on a flower in a sunlit meadow, holliemengert artstyle" \
    "sd15_comic_diffusion_holliemengert"

# James Daly — editorial ink illustration
enqueue \
    "configs/sd15_comic_diffusion_jamesdaly.json" \
    "a detective in a trenchcoat standing in a rain-soaked city alley, jamesdaly artstyle" \
    "sd15_comic_diffusion_jamesdaly"

# Mario Alberti — European superhero comics
enqueue \
    "configs/sd15_comic_diffusion_marioalberti.json" \
    "a superhero leaping over buildings against a dramatic sky, marioalberti artstyle" \
    "sd15_comic_diffusion_marioalberti"

# Pepe Larraz — Marvel-style superhero action
enqueue \
    "configs/sd15_comic_diffusion_pepelarraz.json" \
    "X-Men battle scene with lightning and energy blasts, pepelarraz artstyle" \
    "sd15_comic_diffusion_pepelarraz"

echo

# ===========================================================================
# Stable Diffusion 2.1
# ===========================================================================

echo "── Stable Diffusion 2.1 ────────────────────────────────────"

# SD 2.1 Coloring Book LoRA
enqueue \
    "configs/sd21_coloringbook_redmond_lora.json" \
    "a cute owl perched on a branch, Coloring Book, ColoringBookAF, clean bold outlines, white fill areas" \
    "sd21_coloringbook_redmond_lora"

echo

# ---------------------------------------------------------------------------
