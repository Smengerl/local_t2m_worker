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
| [`create_examples_flux.sh`](create_examples_flux.sh) | `flux`, `flux2_klein` | 8 | FLUX.1 and FLUX.2 configs |
| [`create_examples_sd.sh`](create_examples_sd.sh) | `sd` | 12 | SD 1.5 · SD 2.1 |
| [`create_examples_sd3.sh`](create_examples_sd3.sh) | `sd3` | 1 | Stable Diffusion 3 |
| [`create_examples_sdxl.sh`](create_examples_sdxl.sh) | `sdxl` | 11 | SDXL base and LoRA configs |
| [`create_examples_zimage.sh`](create_examples_zimage.sh) | `zimage` | 3 | Z-Image Turbo — base + Classic Painting LoRA + 1950s American Dream LoRA |
| [`create_examples_all.sh`](create_examples_all.sh) | all | 35 | Orchestrator — runs all scripts above |
| [`_helper.sh`](_helper.sh) | — | — | Shared helper (source only, not executable) |

---

## Usage

### Run all examples at once

```bash
# Run all scripts in sequence
./examples/create_examples_all.sh


### Run a single pipeline

```bash
# FLUX only
./examples/create_examples_flux.sh


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

# CLI status check
./scripts/health_check.sh

# Web dashboard (only if queueing mode is used, not direct generation. Queueing mode can be set active in _helper.sh)
open http://localhost:8000

```

---

## Configs covered (35 jobs)

Images appear automatically once the scripts have been run and the images generated.

### FLUX (`create_examples_flux.sh`)

<div style="display: flex; flex-wrap: wrap; gap: 24px;">
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="flux_schnell.png" alt="flux_schnell" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a misty forest at dawn, sunrays filtering through tall pine trees, volumetric light, photorealistic</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/flux_schnell.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="flux_dev.png" alt="flux_dev" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a majestic snow-capped mountain reflected in a crystal-clear alpine lake, golden hour, hyperrealistic photography</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/flux_dev.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="flux2_klein_4b.png" alt="flux2_klein_4b" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a golden sunset over mountain peaks, warm light painting the clouds in shades of orange and purple, cinematic</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/flux2_klein_4b.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="flux_cute_comic_lora.png" alt="flux_cute_comic_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a brave little fox on an adventure through an enchanted forest, in the style of TOK</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/flux_cute_comic_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="flux_miniature_people_lora.png" alt="flux_miniature_people_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>miniature people hiking on a sandwich used as a mountain trail, macro photography, shallow depth of field</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/flux_miniature_people_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="flux_dev_wong_kar_wai_fallen_angels_lora.png" alt="flux_dev_wong_kar_wai_fallen_angels_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>WKW style, tilted view of a woman with long dark hair standing in a neon-lit subway station, rain streaking the windows, moody neo-noir atmosphere, vivid green and red neon reflections</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/flux_dev_wong_kar_wai_fallen_angels_lora.json</code></div>
  </div>
 </div>
</div>

### Stable Diffusion 1.5 (`create_examples_sd.sh`)

<div style="display: flex; flex-wrap: wrap; gap: 24px;">
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_default.png" alt="sd15_default" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a photorealistic portrait of a young woman with freckles, soft natural window light, shallow depth of field</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_default.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_dreamshaper8.png" alt="sd15_dreamshaper8" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>fantasy portrait of an elf warrior in enchanted armour, dramatic rim lighting, detailed digital painting</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_dreamshaper8.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_realistic_vision_v6.png" alt="sd15_realistic_vision_v6" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>RAW photo, portrait of a young woman with auburn hair, soft studio lighting, sharp focus, 85mm lens</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_realistic_vision_v6.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_inkpunk_lora.png" alt="sd15_inkpunk_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>nvinkpunk portrait of a samurai warrior, ink brush strokes, cyberpunk city in the background</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_inkpunk_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_pixel_art_lora.png" alt="sd15_pixel_art_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>pixel art dragon on a mountain peak, 16-bit retro game style, sunset colours</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_pixel_art_lora.json</code></div>
  </div>
 </div>
</div>

### Comic Diffusion — artist styles (`create_examples_sd.sh`)

<div style="display: flex; flex-wrap: wrap; gap: 24px;">
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_comic_diffusion_andreasrocha.png" alt="sd15_comic_diffusion_andreasrocha" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a castle on a cliff overlooking a misty valley at sunset, andreasrocha artstyle</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_comic_diffusion_andreasrocha.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_comic_diffusion_charliebo.png" alt="sd15_comic_diffusion_charliebo" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a superhero leaping across rooftops in a neon city at night, charliebo artstyle</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_comic_diffusion_charliebo.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_comic_diffusion_holliemengert.png" alt="sd15_comic_diffusion_holliemengert" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a fairy sitting on a flower in a sunlit meadow, holliemengert artstyle</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_comic_diffusion_holliemengert.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_comic_diffusion_jamesdaly.png" alt="sd15_comic_diffusion_jamesdaly" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a detective in a trenchcoat standing in a rain-soaked city alley, jamesdaly artstyle</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_comic_diffusion_jamesdaly.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_comic_diffusion_marioalberti.png" alt="sd15_comic_diffusion_marioalberti" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a superhero leaping over buildings against a dramatic sky, marioalberti artstyle</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_comic_diffusion_marioalberti.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd15_comic_diffusion_pepelarraz.png" alt="sd15_comic_diffusion_pepelarraz" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>X-Men battle scene with lightning and energy blasts, pepelarraz artstyle</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd15_comic_diffusion_pepelarraz.json</code></div>
  </div>
 </div>
</div>

### Stable Diffusion 2.1 (`create_examples_sd.sh`)

<div style="display: flex; flex-wrap: wrap; gap: 24px;">
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd21_coloringbook_redmond_lora.png" alt="sd21_coloringbook_redmond_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a cute owl perched on a branch, Coloring Book, ColoringBookAF, clean bold outlines, white fill areas</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd21_coloringbook_redmond_lora.json</code></div>
  </div>
 </div>
</div>

### Stable Diffusion 3 (`create_examples_sd3.sh`)

<div style="display: flex; flex-wrap: wrap; gap: 24px;">
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sd3_medium.png" alt="sd3_medium" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a vibrant street market in Marrakech at golden hour, detailed textures, rich colours, cinematic composition</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sd3_medium.json</code></div>
  </div>
 </div>
</div>

### SDXL (`create_examples_sdxl.sh`)

<div style="display: flex; flex-wrap: wrap; gap: 24px;">
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_turbo.png" alt="sdxl_turbo" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a photorealistic tabby cat sitting on a windowsill in warm afternoon light</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_turbo.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_hypersd_lora.png" alt="sdxl_hypersd_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a photorealistic portrait of a woman with green eyes, professional studio lighting, bokeh background</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_hypersd_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_analog_redmond_lora.png" alt="sdxl_analog_redmond_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>AnalogRedmAF portrait of a young man at golden hour, film grain, warm tones, 35mm photograph</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_analog_redmond_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_bandw_manga_lora.png" alt="sdxl_bandw_manga_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a boy in a sailor school uniform standing on a rooftop at sunset, manga illustration, bold ink lines</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_bandw_manga_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_graffiti_lora.png" alt="sdxl_graffiti_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>graarg graffiti mural of a roaring lion on a brick wall, vibrant spray-paint colours</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_graffiti_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_ikea_lora.png" alt="sdxl_ikea_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a bicycle, flat minimalist assembly diagram illustration, black lines on white background</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_ikea_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_littletinies_lora.png" alt="sdxl_littletinies_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a tiny witch child riding a broomstick over a moonlit village, soft pastel colours, hand-drawn style</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_littletinies_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_papercut_lora.png" alt="sdxl_papercut_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>papercut forest with deer and mountains, intricate layered paper silhouettes, white background</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_papercut_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_pokemon_sprite_lora.png" alt="sdxl_pokemon_sprite_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a Pokémon trainer sprite in pixel art style, gen1 palette, facing forward</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_pokemon_sprite_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_storyboard_sketch_lora.png" alt="sdxl_storyboard_sketch_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>storyboard sketch of a hero running through a collapsing building, dynamic camera angle, grayscale pencil</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_storyboard_sketch_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="sdxl_watercolor_lora.png" alt="sdxl_watercolor_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a mountain landscape at sunset with a winding river, loose watercolour painting, soft bleeding edges</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/sdxl_watercolor_lora.json</code></div>
  </div>
 </div>
</div>

### Z-Image (`create_examples_zimage.sh`)

<div style="display: flex; flex-wrap: wrap; gap: 24px;">
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="zimage_turbo.png" alt="zimage_turbo" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>a photorealistic portrait of a woman in soft evening light, warm golden background, sharp focus</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/zimage_turbo.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="zimage_turbo_classic_painting_lora.png" alt="zimage_turbo_classic_painting_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>class1cpa1nt classic oil painting of a pensive woman in 17th-century attire, soft candlelight, dark textured background, catchlight details in her eyes, photorealistic brushwork</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/zimage_turbo_classic_painting_lora.json</code></div>
  </div>
 </div>
 <div style="flex: 0 1 320px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 2px 8px #0001; overflow: hidden; background: transparent;">
  <img src="zimage_turbo_1950s_american_dream_lora.png" alt="zimage_turbo_1950s_american_dream_lora" style="width: 100%; display: block; aspect-ratio: 1/1; object-fit: cover;">
  <div style="padding: 16px;">
   <div style="font-size: 1em; margin-bottom: 12px;"><strong>Prompt:</strong><br>5os4m3r1c4n4, 1950s, painting, a painting of a cheerful american family at a drive-in diner, pastel colours, chrome details, sunny afternoon, retro Americana</div>
   <hr>
   <div style="text-align: center; font-family: monospace; font-size: 0.95em; color: #444;"><code>configs/zimage_turbo_1950s_american_dream_lora.json</code></div>
  </div>
 </div>
</div>

> **NSFW configs** (in `configs/nsfw/`) are intentionally excluded.
