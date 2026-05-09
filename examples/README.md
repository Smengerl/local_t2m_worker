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

## Usage

### Run all examples at once

```bash
# Run all scripts in sequence
./examples/create_examples_all.sh
```

### Run a single pipeline

```bash
# FLUX only
./examples/create_examples_flux.sh
```

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

# **Examples:**

**Examples:**

### FLUX (`create_examples_flux.sh`)

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="flux_schnell.png" width="240" vspace="0" hspace="0" alt="flux_schnell"></td>
<td width="8"></td>
<td width="240"><img src="flux_cute_comic_lora.png" width="240" vspace="0" hspace="0" alt="flux_cute_comic_lora"></td>
<td width="8"></td>
<td width="240"><img src="flux_miniature_people_lora.png" width="240" vspace="0" hspace="0" alt="flux_miniature_people_lora"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>a misty forest at dawn, sunrays filtering through tall pine trees, volumetric light, photorealistic<br><br></td>
<td></td>
<td align="center" valign="top"><br>a brave little fox on an adventure through an enchanted forest, in the style of TOK<br><br></td>
<td></td>
<td align="center" valign="top"><br>miniature people hiking on a sandwich used as a mountain trail, macro photography, shallow depth of field<br><br></td>
</tr>
<tr>
<td align="center"><code>flux_schnell.json</code></td>
<td></td>
<td align="center"><code>flux_cute_comic_lora.json</code></td>
<td></td>
<td align="center"><code>flux_miniature_people_lora.json</code></td>
</tr>
</table>

<br>

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="flux_dev.png" width="240" vspace="0" hspace="0" alt="flux_dev"></td>
<td width="8"></td>
<td width="240"><img src="flux_dev_wong_kar_wai_fallen_angels_lora.png" width="240" vspace="0" hspace="0" alt="flux_dev_wong_kar_wai_fallen_angels_lora"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>a majestic snow-capped mountain reflected in a crystal-clear alpine lake, golden hour, hyperrealistic photography<br><br></td>
<td></td>
<td align="center" valign="top"><br>WKW style, tilted view of a woman with long dark hair standing in a neon-lit subway station, rain streaking the windows, moody neo-noir atmosphere, vivid green and red neon reflections<br><br></td>
</tr>
<tr>
<td align="center"><code>flux_dev.json</code></td>
<td></td>
<td align="center"><code>flux_dev_wong_kar_wai_fallen_angels_lora.json</code></td>
</tr>
</table>

<br>

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="flux2_klein_4b.png" width="240" vspace="0" hspace="0" alt="flux2_klein_4b"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>a golden sunset over mountain peaks, warm light painting the clouds in shades of orange and purple, cinematic<br><br></td>
</tr>
<tr>
<td align="center"><code>flux2_klein_4b.json</code></td>
</tr>
</table>

<br>

### Stable Diffusion 1.5 (`create_examples_sd.sh`)

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="sd15_default.png" width="240" vspace="0" hspace="0" alt="sd15_default"></td>
<td width="8"></td>
<td width="240"><img src="sd15_inkpunk_lora.png" width="240" vspace="0" hspace="0" alt="sd15_inkpunk_lora"></td>
<td width="8"></td>
<td width="240"><img src="sd15_pixel_art_lora.png" width="240" vspace="0" hspace="0" alt="sd15_pixel_art_lora"></td>
<td width="8"></td>
<td width="240"><img src="sd15_elden_ring.png" width="240" vspace="0" hspace="0" alt="sd15_elden_ring"></td>
<td width="8"></td>
<td width="240"><img src="sd15_comic_diffusion_andreasrocha.png" width="240" vspace="0" hspace="0" alt="sd15_comic_diffusion_andreasrocha"></td>
<td width="8"></td>
<td width="240"><img src="sd15_comic_diffusion_charliebo.png" width="240" vspace="0" hspace="0" alt="sd15_comic_diffusion_charliebo"></td>
<td width="8"></td>
<td width="240"><img src="sd15_comic_diffusion_holliemengert.png" width="240" vspace="0" hspace="0" alt="sd15_comic_diffusion_holliemengert"></td>
<td width="8"></td>
<td width="240"><img src="sd15_comic_diffusion_jamesdaly.png" width="240" vspace="0" hspace="0" alt="sd15_comic_diffusion_jamesdaly"></td>
<td width="8"></td>
<td width="240"><img src="sd15_comic_diffusion_marioalberti.png" width="240" vspace="0" hspace="0" alt="sd15_comic_diffusion_marioalberti"></td>
<td width="8"></td>
<td width="240"><img src="sd15_comic_diffusion_pepelarraz.png" width="240" vspace="0" hspace="0" alt="sd15_comic_diffusion_pepelarraz"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">FINETUNED</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>a photorealistic portrait of a young woman with freckles, soft natural window light, shallow depth of field<br><br></td>
<td></td>
<td align="center" valign="top"><br>nvinkpunk portrait of a samurai warrior, ink brush strokes, cyberpunk city in the background<br><br></td>
<td></td>
<td align="center" valign="top"><br>pixel art dragon on a mountain peak, 16-bit retro game style, sunset colours<br><br></td>
<td></td>
<td align="center" valign="top"><br>elden ring style, tarnished warrior in ornate golden armor standing before the Erdtree at dusk<br><br></td>
<td></td>
<td align="center" valign="top"><br>a castle on a cliff overlooking a misty valley at sunset, andreasrocha artstyle<br><br></td>
<td></td>
<td align="center" valign="top"><br>a superhero leaping across rooftops in a neon city at night, charliebo artstyle<br><br></td>
<td></td>
<td align="center" valign="top"><br>a fairy sitting on a flower in a sunlit meadow, holliemengert artstyle<br><br></td>
<td></td>
<td align="center" valign="top"><br>a detective in a trenchcoat standing in a rain-soaked city alley, jamesdaly artstyle<br><br></td>
<td></td>
<td align="center" valign="top"><br>a superhero leaping over buildings against a dramatic sky, marioalberti artstyle<br><br></td>
<td></td>
<td align="center" valign="top"><br>X-Men battle scene with lightning and energy blasts, pepelarraz artstyle<br><br></td>
</tr>
<tr>
<td align="center"><code>sd15_default.json</code></td>
<td></td>
<td align="center"><code>sd15_inkpunk_lora.json</code></td>
<td></td>
<td align="center"><code>sd15_pixel_art_lora.json</code></td>
<td></td>
<td align="center"><code>sd15_comic_diffusion_andreasrocha.json</code></td>
<td></td>
<td align="center"><code>sd15_comic_diffusion_charliebo.json</code></td>
<td></td>
<td align="center"><code>sd15_comic_diffusion_holliemengert.json</code></td>
<td></td>
<td align="center"><code>sd15_comic_diffusion_jamesdaly.json</code></td>
<td></td>
<td align="center"><code>sd15_comic_diffusion_marioalberti.json</code></td>
<td></td>
<td align="center"><code>sd15_comic_diffusion_pepelarraz.json</code></td>
</tr>
</table>

<br>

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="sd15_dreamshaper8.png" width="240" vspace="0" hspace="0" alt="sd15_dreamshaper8"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>fantasy portrait of an elf warrior in enchanted armour, dramatic rim lighting, detailed digital painting<br><br></td>
</tr>
<tr>
<td align="center"><code>sd15_dreamshaper8.json</code></td>
</tr>
</table>

<br>

### Stable Diffusion 2.1 (`create_examples_sd.sh`)

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="sd21_default.png" width="240" vspace="0" hspace="0" alt="sd21_default"></td>
<td width="8"></td>
<td width="240"><img src="sd21_coloringbook_redmond_lora.png" width="240" vspace="0" hspace="0" alt="sd21_coloringbook_redmond_lora"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>a solitary lighthouse on a rocky cliff at dusk, dramatic storm clouds, crashing waves, cinematic lighting<br><br></td>
<td></td>
<td align="center" valign="top"><br>a cute owl perched on a branch, Coloring Book, ColoringBookAF, clean bold outlines, white fill areas<br><br></td>
</tr>
<tr>
<td align="center"><code>sd21_default.json</code></td>
<td></td>
<td align="center"><code>sd21_coloringbook_redmond_lora.json</code></td>
</tr>
</table>

<br>

### Stable Diffusion 3 (`create_examples_sd3.sh`)

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="sd3_medium.png" width="240" vspace="0" hspace="0" alt="sd3_medium"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>a vibrant street market in Marrakech at golden hour, detailed textures, rich colours, cinematic composition<br><br></td>
</tr>
<tr>
<td align="center"><code>sd3_medium.json</code></td>
</tr>
</table>

<br>

### SDXL (`create_examples_sdxl.sh`)

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="sdxl_turbo.png" width="240" vspace="0" hspace="0" alt="sdxl_turbo"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_hypersd_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_hypersd_lora"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_analog_redmond_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_analog_redmond_lora"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_bandw_manga_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_bandw_manga_lora"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_graffiti_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_graffiti_lora"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_littletinies_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_littletinies_lora"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_papercut_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_papercut_lora"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_pokemon_sprite_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_pokemon_sprite_lora"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_storyboard_sketch_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_storyboard_sketch_lora"></td>
<td width="8"></td>
<td width="240"><img src="sdxl_watercolor_lora.png" width="240" vspace="0" hspace="0" alt="sdxl_watercolor_lora"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>a photorealistic tabby cat sitting on a windowsill in warm afternoon light<br><br></td>
<td></td>
<td align="center" valign="top"><br>hyperrealistic close-up of an elderly fisherman with deeply weathered skin and piercing blue eyes, dramatic rembrandt lighting, sharp focus on facial texture<br><br></td>
<td></td>
<td align="center" valign="top"><br>AnalogRedmAF portrait of a young man at golden hour, film grain, warm tones, 35mm photograph<br><br></td>
<td></td>
<td align="center" valign="top"><br>a boy in a sailor school uniform standing on a rooftop at sunset, manga illustration, bold ink lines<br><br></td>
<td></td>
<td align="center" valign="top"><br>graarg graffiti mural of a roaring lion on a brick wall, vibrant spray-paint colours<br><br></td>
<td></td>
<td align="center" valign="top"><br>a tiny witch child riding a broomstick over a moonlit village, soft pastel colours, hand-drawn style<br><br></td>
<td></td>
<td align="center" valign="top"><br>papercut forest with deer and mountains, intricate layered paper silhouettes, white background<br><br></td>
<td></td>
<td align="center" valign="top"><br>a Pokémon trainer sprite in pixel art style, gen1 palette, facing forward<br><br></td>
<td></td>
<td align="center" valign="top"><br>storyboard sketch of a hero running through a collapsing building, dynamic camera angle, grayscale pencil<br><br></td>
<td></td>
<td align="center" valign="top"><br>a mountain landscape at sunset with a winding river, loose watercolour painting, soft bleeding edges<br><br></td>
</tr>
<tr>
<td align="center"><code>sdxl_turbo.json</code></td>
<td></td>
<td align="center"><code>sdxl_hypersd_lora.json</code></td>
<td></td>
<td align="center"><code>sdxl_analog_redmond_lora.json</code></td>
<td></td>
<td align="center"><code>sdxl_bandw_manga_lora.json</code></td>
<td></td>
<td align="center"><code>sdxl_graffiti_lora.json</code></td>
<td></td>
<td align="center"><code>sdxl_littletinies_lora.json</code></td>
<td></td>
<td align="center"><code>sdxl_papercut_lora.json</code></td>
<td></td>
<td align="center"><code>sdxl_pokemon_sprite_lora.json</code></td>
<td></td>
<td align="center"><code>sdxl_storyboard_sketch_lora.json</code></td>
<td></td>
<td align="center"><code>sdxl_watercolor_lora.json</code></td>
</tr>
</table>

<br>

### Z-Image (`create_examples_zimage.sh`)

<table cellpadding="0" cellspacing="0" border="0">
<tr>
<td width="240"><img src="zimage_turbo.png" width="240" vspace="0" hspace="0" alt="zimage_turbo"></td>
<td width="8"></td>
<td width="240"><img src="zimage_turbo_classic_painting_lora.png" width="240" vspace="0" hspace="0" alt="zimage_turbo_classic_painting_lora"></td>
<td width="8"></td>
<td width="240"><img src="zimage_turbo_1950s_american_dream_lora.png" width="240" vspace="0" hspace="0" alt="zimage_turbo_1950s_american_dream_lora"></td>
</tr>
<tr>
<td align="center" bgcolor="#dbeafe"><b><font color="#000000">BASE MODEL</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
<td></td>
<td align="center" bgcolor="#1e40af"><b><font color="#ffffff">LORA</font></b></td>
</tr>
<tr>
<td align="center" valign="top"><br>a photorealistic portrait of a woman in soft evening light, warm golden background, sharp focus<br><br></td>
<td></td>
<td align="center" valign="top"><br>class1cpa1nt classic oil painting of a pensive woman in 17th-century attire, soft candlelight, dark textured background, catchlight details in her eyes, photorealistic brushwork<br><br></td>
<td></td>
<td align="center" valign="top"><br>5os4m3r1c4n4, 1950s, painting, a painting of a cheerful american family at a drive-in diner, pastel colours, chrome details, sunny afternoon, retro Americana<br><br></td>
</tr>
<tr>
<td align="center"><code>zimage_turbo.json</code></td>
<td></td>
<td align="center"><code>zimage_turbo_classic_painting_lora.json</code></td>
<td></td>
<td align="center"><code>zimage_turbo_1950s_american_dream_lora.json</code></td>
</tr>
</table>
