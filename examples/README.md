# Examples

Ready-to-run shell scripts that enqueue one showcase generation job per config via the CLI queue mode (`scripts/run.sh --queue`).

These scripts use the CLI queue, not the REST API. You do **not** need to start the web server for job submission—only the batch worker must be running. If you do not start the server, you can still monitor progress and job status using the CLI tool `./scripts/health_check.sh`.

Use these scripts to verify that all configs work end-to-end and to produce a visual comparison of every supported style.

> **Prerequisite:** The batch worker/server must be running before you execute any script:
>
> ```bash
> ./scripts/run_batch_server.sh
> ```

---

## Scripts

| Script | Pipeline backend | Jobs | Description |
|---|---|---|---|
| [`create_examples_flux.sh`](create_examples_flux.sh) | `flux`, `flux2_klein` | 7 | FLUX.1 and FLUX.2 configs |
| [`create_examples_sd.sh`](create_examples_sd.sh) | `sd` | 12 | SD 1.5 · Comic Diffusion · SD 2.1 |
| [`create_examples_sd3.sh`](create_examples_sd3.sh) | `sd3` | 1 | Stable Diffusion 3 |
| [`create_examples_sdxl.sh`](create_examples_sdxl.sh) | `sdxl` | 11 | SDXL base and LoRA configs |
| [`create_examples_zimage.sh`](create_examples_zimage.sh) | `zimage` | 1 | Z-Image Turbo |
| [`create_examples_all.sh`](create_examples_all.sh) | all | 32 | Orchestrator — runs all scripts above |
| [`_helper.sh`](_helper.sh) | — | — | Shared helper (source only, not executable) |

---

## Usage

### Run all examples at once

```bash
# Enqueue all 32 showcase jobs (default server: http://localhost:8000)
./examples/create_examples_all.sh

# Custom server port
BASE_URL=http://localhost:9000 ./examples/create_examples_all.sh
```

### Run a single pipeline

```bash
# FLUX only
./examples/create_examples_flux.sh

# SDXL only, custom server
BASE_URL=http://localhost:9000 ./examples/create_examples_sdxl.sh
```

Each per-pipeline script works standalone **and** can be sourced by `create_examples_all.sh` without a duplicate server check.

---

## How it works

1. `_helper.sh` is sourced by every script — it provides `enqueue()`, `check_server()`, `BASE_URL`, and `OUT_DIR`.
2. Each per-pipeline script enqueues its jobs and works standalone.
3. `create_examples_all.sh` performs one server check then sources all per-pipeline scripts in sequence.
4. Each job uses the config's built-in default parameters (no parameter overrides).
5. A `✅ / ❌` summary line is printed per job.

---

## Output

All generated images land in this directory (`examples/`), named after their config.

Monitor progress while jobs run:

```bash
# Web dashboard
open http://localhost:8000

# CLI status check
./scripts/health_check.sh
```

---

## Configs covered (32 jobs)

Images appear automatically once the scripts have been run and the images generated.

### FLUX (`create_examples_flux.sh`)

| Image | Config | Prompt |
|---|---|---|
| <img src="flux_schnell.png" alt="flux_schnell" width="500"> | `flux_schnell` | a misty forest at dawn, sunrays filtering through tall pine trees, volumetric light, photorealistic |
| <img src="flux_dev.png" alt="flux_dev" width="500"> | `flux_dev` | a majestic snow-capped mountain reflected in a crystal-clear alpine lake, golden hour, hyperrealistic photography |
| <img src="flux2_klein_4b.png" alt="flux2_klein_4b" width="500"> | `flux2_klein_4b` | a golden sunset over mountain peaks, warm light painting the clouds in shades of orange and purple, cinematic |
| <img src="flux_cute_comic_lora.png" alt="flux_cute_comic_lora" width="500"> | `flux_cute_comic_lora` | a brave little fox on an adventure through an enchanted forest, in the style of TOK |
| <img src="flux_dev_cute_comic_lora.png" alt="flux_dev_cute_comic_lora" width="500"> | `flux_dev_cute_comic_lora` | a cheerful robot baking cookies in a cosy kitchen, in the style of TOK |
| <img src="flux_miniature_people_lora.png" alt="flux_miniature_people_lora" width="500"> | `flux_miniature_people_lora` | miniature people hiking on a sandwich used as a mountain trail, macro photography, shallow depth of field |
| <img src="flux_dev_miniature_people_lora.png" alt="flux_dev_miniature_people_lora" width="500"> | `flux_dev_miniature_people_lora` | miniature people camping next to a candle flame at night, macro photography, bokeh |

### Stable Diffusion 1.5 (`create_examples_sd.sh`)

| Image | Config | Prompt |
|---|---|---|
| <img src="sd15_default.png" alt="sd15_default" width="500"> | `sd15_default` | a photorealistic portrait of a young woman with freckles, soft natural window light, shallow depth of field |
| <img src="sd15_dreamshaper8.png" alt="sd15_dreamshaper8" width="500"> | `sd15_dreamshaper8` | fantasy portrait of an elf warrior in enchanted armour, dramatic rim lighting, detailed digital painting |
| <img src="sd15_realistic_vision_v6.png" alt="sd15_realistic_vision_v6" width="500"> | `sd15_realistic_vision_v6` | RAW photo, portrait of a young woman with auburn hair, soft studio lighting, sharp focus, 85mm lens |
| <img src="sd15_inkpunk_lora.png" alt="sd15_inkpunk_lora" width="500"> | `sd15_inkpunk_lora` | nvinkpunk portrait of a samurai warrior, ink brush strokes, cyberpunk city in the background |
| <img src="sd15_pixel_art_lora.png" alt="sd15_pixel_art_lora" width="500"> | `sd15_pixel_art_lora` | pixel art dragon on a mountain peak, 16-bit retro game style, sunset colours |

### Comic Diffusion — artist styles (`create_examples_sd.sh`)

| Image | Config | Prompt |
|---|---|---|
| <img src="sd15_comic_diffusion_andreasrocha.png" alt="sd15_comic_diffusion_andreasrocha" width="500"> | `sd15_comic_diffusion_andreasrocha` | a castle on a cliff overlooking a misty valley at sunset, andreasrocha artstyle |
| <img src="sd15_comic_diffusion_charliebo.png" alt="sd15_comic_diffusion_charliebo" width="500"> | `sd15_comic_diffusion_charliebo` | a superhero leaping across rooftops in a neon city at night, charliebo artstyle |
| <img src="sd15_comic_diffusion_holliemengert.png" alt="sd15_comic_diffusion_holliemengert" width="500"> | `sd15_comic_diffusion_holliemengert` | a fairy sitting on a flower in a sunlit meadow, holliemengert artstyle |
| <img src="sd15_comic_diffusion_jamesdaly.png" alt="sd15_comic_diffusion_jamesdaly" width="500"> | `sd15_comic_diffusion_jamesdaly` | a detective in a trenchcoat standing in a rain-soaked city alley, jamesdaly artstyle |
| <img src="sd15_comic_diffusion_marioalberti.png" alt="sd15_comic_diffusion_marioalberti" width="500"> | `sd15_comic_diffusion_marioalberti` | a superhero leaping over buildings against a dramatic sky, marioalberti artstyle |
| <img src="sd15_comic_diffusion_pepelarraz.png" alt="sd15_comic_diffusion_pepelarraz" width="500"> | `sd15_comic_diffusion_pepelarraz` | X-Men battle scene with lightning and energy blasts, pepelarraz artstyle |

### Stable Diffusion 2.1 (`create_examples_sd.sh`)

| Image | Config | Prompt |
|---|---|---|
| <img src="sd21_coloringbook_redmond_lora.png" alt="sd21_coloringbook_redmond_lora" width="500"> | `sd21_coloringbook_redmond_lora` | a cute owl perched on a branch, Coloring Book, ColoringBookAF, clean bold outlines, white fill areas |

### Stable Diffusion 3 (`create_examples_sd3.sh`)

| Image | Config | Prompt |
|---|---|---|
| <img src="sd3_medium.png" alt="sd3_medium" width="500"> | `sd3_medium` | a vibrant street market in Marrakech at golden hour, detailed textures, rich colours, cinematic composition |

### SDXL (`create_examples_sdxl.sh`)

| Image | Config | Prompt |
|---|---|---|
| <img src="sdxl_turbo.png" alt="sdxl_turbo" width="500"> | `sdxl_turbo` | a photorealistic tabby cat sitting on a windowsill in warm afternoon light |
| <img src="sdxl_hypersd_lora.png" alt="sdxl_hypersd_lora" width="500"> | `sdxl_hypersd_lora` | a photorealistic portrait of a woman with green eyes, professional studio lighting, bokeh background |
| <img src="sdxl_analog_redmond_lora.png" alt="sdxl_analog_redmond_lora" width="500"> | `sdxl_analog_redmond_lora` | AnalogRedmAF portrait of a young man at golden hour, film grain, warm tones, 35mm photograph |
| <img src="sdxl_bandw_manga_lora.png" alt="sdxl_bandw_manga_lora" width="500"> | `sdxl_bandw_manga_lora` | a boy in a sailor school uniform standing on a rooftop at sunset, manga illustration, bold ink lines |
| <img src="sdxl_graffiti_lora.png" alt="sdxl_graffiti_lora" width="500"> | `sdxl_graffiti_lora` | graarg graffiti mural of a roaring lion on a brick wall, vibrant spray-paint colours |
| <img src="sdxl_ikea_lora.png" alt="sdxl_ikea_lora" width="500"> | `sdxl_ikea_lora` | a bicycle, flat minimalist assembly diagram illustration, black lines on white background |
| <img src="sdxl_littletinies_lora.png" alt="sdxl_littletinies_lora" width="500"> | `sdxl_littletinies_lora` | a tiny witch child riding a broomstick over a moonlit village, soft pastel colours, hand-drawn style |
| <img src="sdxl_papercut_lora.png" alt="sdxl_papercut_lora" width="500"> | `sdxl_papercut_lora` | papercut forest with deer and mountains, intricate layered paper silhouettes, white background |
| <img src="sdxl_pokemon_sprite_lora.png" alt="sdxl_pokemon_sprite_lora" width="500"> | `sdxl_pokemon_sprite_lora` | a Pokémon trainer sprite in pixel art style, gen1 palette, facing forward |
| <img src="sdxl_storyboard_sketch_lora.png" alt="sdxl_storyboard_sketch_lora" width="500"> | `sdxl_storyboard_sketch_lora` | storyboard sketch of a hero running through a collapsing building, dynamic camera angle, grayscale pencil |
| <img src="sdxl_watercolor_lora.png" alt="sdxl_watercolor_lora" width="500"> | `sdxl_watercolor_lora` | a mountain landscape at sunset with a winding river, loose watercolour painting, soft bleeding edges |

### Z-Image (`create_examples_zimage.sh`)

| Image | Config | Prompt |
|---|---|---|
| <img src="zimage_turbo.png" alt="zimage_turbo" width="500"> | `zimage_turbo` | a photorealistic portrait of a woman in soft evening light, warm golden background, sharp focus |

> **NSFW configs** (in `configs/nsfw/`) are intentionally excluded.
