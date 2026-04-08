# Configuration Reference

Config files are JSON files passed via `--config` / `-c`. They describe which model, LoRA, adapter, and generation parameters to use. CLI flags always override config values; config values override built-in defaults.

## Config file structure

```json
{
    "description": "Human-readable label shown at startup",
    "pipeline_type": "sdxl",
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "lora_id": "linoyts/lora-xl-graffiti-0.0001-5e-05-1000-1-None",
    "lora_scale": 0.9,
    "weight_name": null,
    "trigger_word": "graarg graffiti",
    "adapter_id": null,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "sequential_cpu_offload": true,
    "seed": null
}
```

`_comment*` keys are ignored by the parser and can be used for inline documentation.

## Available keys

| Key | Default | Description |
|---|---|---|
| `pipeline_type` | **(required)** | `"sd"` / `"sdxl"` / `"sd3"` / `"flux"` / `"zimage"` / `"qwen"` — selects the backend |
| `model_id` | SD 1.5 repo | HuggingFace repo ID or local path of the base model |
| `description` | `null` | Label shown in logs and the web dashboard |
| `adapter_id` | `null` | HF repo ID / local path for an adapter (ControlNet, refiner, …) |
| `lora_id` | `null` | HF repo ID / local path for LoRA weights |
| `lora_scale` | `0.9` | LoRA blending strength (`0.0`–`1.0`) |
| `weight_name` | `null` | Specific `.safetensors` filename inside a multi-file LoRA or base-model repo (required for monorepos like `ByteDance/Hyper-SD`) |
| `trigger_word` | `null` | Token required by the LoRA. Automatically prepended to the prompt with a warning if missing. |
| `num_inference_steps` | `30` | Denoising steps — more = better quality, slower |
| `guidance_scale` | `7.5` | CFG scale — how strongly the model follows the prompt; set to `0.0` for distilled models (SDXL-Turbo, FLUX-schnell, Z-Image-Turbo) |
| `width` / `height` | `1024` | Output resolution in pixels |
| `sequential_cpu_offload` | `false` | Offloads sub-modules to CPU between operations, cutting peak VRAM ~50 %. Recommended for SDXL and FLUX on ≤ 16 GB RAM. |
| `output_dir` | `"outputs"` | Directory for generated images |
| `cache_dir` | `"models"` | Directory for downloaded model weights |
| `seed` | `null` | Fixed RNG seed for reproducible results. `null` = random. |
| `true_cfg_scale` | `null` | Secondary guidance scale used by some backends (e.g. Qwen-Image) |

## Adding a new config

1. Copy an existing config that uses the same `pipeline_type`.
2. Set `model_id` and optionally `lora_id`.
3. Adjust `num_inference_steps`, `guidance_scale`, `width`, `height` per the model card.
4. Enable `sequential_cpu_offload: true` for any SDXL / FLUX / SD3 model on Mac.
5. Save to `configs/your_name.json` — it is picked up automatically by the web dashboard and `run.sh`.

---

## Existing configs

### Stable Diffusion 1.5 (`pipeline_type: "sd"`)

| File | LoRA / adapter | Trigger word | Size | Description |
|---|---|---|---|---|
| `sd15_default.json` | — | — | 512×512 | SD 1.5 base — general-purpose text-to-image, ~4 GB RAM |
| `sd15_dreamshaper8.json` | — | — | 512×512 | DreamShaper v8 — versatile fine-tune for fantasy, portraits, stylised realism |
| `sd15_realistic_vision_v6.json` | — | — | 768×768 | Realistic Vision V6.0 B1 — hyperrealistic portrait / photorealism (noVAE variant) |
| `sd15_inkpunk_lora.json` | — | `nvinkpunk` | 512×512 | Inkpunk Diffusion — ink/punk illustration style (DreamBooth fine-tune) |
| `sd15_pixel_art_lora.json` | SedatAl/pixel-art-LoRa | — | 512×512 | Pixel-art / 16-bit retro style |
| `sd15_lazymix_amateur.json` | — | — | 512×768 | LazyMix+ v4.0 — realistic amateur photography fine-tune |
| `sd15_comic_diffusion_andreasrocha.json` | — | `andreasrocha artstyle` | 512×512 | Comic-Diffusion v2 — Andreas Rocha artstyle |
| `sd15_comic_diffusion_charliebo.json` | — | `charliebo artstyle` | 512×512 | Comic-Diffusion v2 — Charlie Bo artstyle |
| `sd15_comic_diffusion_holliemengert.json` | — | `holliemengert artstyle` | 512×512 | Comic-Diffusion v2 — Hollie Mengert artstyle |
| `sd15_comic_diffusion_jamesdaly.json` | — | `jamesdaly artstyle` | 512×512 | Comic-Diffusion v2 — James Daly artstyle |
| `sd15_comic_diffusion_marioalberti.json` | — | `marioalberti artstyle` | 512×512 | Comic-Diffusion v2 — Mario Alberti artstyle |
| `sd15_comic_diffusion_pepelarraz.json` | — | `pepelarraz artstyle` | 512×512 | Comic-Diffusion v2 — Pepe Larraz artstyle |

### Stable Diffusion 2.1 (`pipeline_type: "sd"`)

| File | LoRA / adapter | Trigger word | Size | Description |
|---|---|---|---|---|
| `sd21_coloringbook_redmond_lora.json` | artificialguybr/coloringbook-redmond-… | `ColoringBookAF` | 512×512 | Coloring-book line-art style LoRA for SD 2.1 |

### Stable Diffusion 3 (`pipeline_type: "sd3"`)

> Gated model — requires a HuggingFace token stored in `.hf_token`.

| File | LoRA / adapter | Trigger word | Size | Description |
|---|---|---|---|---|
| `sd3_medium.json` | — | — | 1024×1024 | SD3-Medium — 2B MMDiT, 28 steps, improved prompt adherence |

### Stable Diffusion XL (`pipeline_type: "sdxl"`)

| File | LoRA / adapter | Trigger word | Size | Description |
|---|---|---|---|---|
| `sdxl_turbo.json` | — | — | 512×512 | SDXL-Turbo — 4-step ADD-distilled; `guidance_scale` must be `0.0` |
| `sdxl_graffiti_lora.json` | linoyts/lora-xl-graffiti-… | `graarg graffiti` | 1024×1024 | Graffiti lettering / mural style |
| `sdxl_littletinies_lora.json` | alvdansen/littletinies | — | 1024×1024 | Soft hand-drawn cartoon style |
| `sdxl_ikea_lora.json` | ostris/ikea-instructions-lora-sdxl | — | 1024×1024 | Flat line-art IKEA assembly-manual style |
| `sdxl_bandw_manga_lora.json` | alvdansen/BandW-Manga | — | 1024×1024 | Bold monochrome manga line-art |
| `sdxl_storyboard_sketch_lora.json` | blink7630/storyboard-sketch | `storyboard sketch of` | 1024×1024 | Grayscale film/TV storyboard style |
| `sdxl_pokemon_sprite_lora.json` | sWizad/pokemon-trainer-sprite-pixelart | — | 768×768 | Pixel-art Pokémon trainer sprite style |
| `sdxl_watercolor_lora.json` | ostris/watercolor_style_lora_sdxl | — | 1024×1024 | Loose watercolor painting — soft edges, visible brush strokes |
| `sdxl_analog_redmond_lora.json` | artificialguybr/analogredmond-v2 | `AnalogRedmAF` | 1024×1024 | Analog film photography — grain, warm tones, vintage look |
| `sdxl_papercut_lora.json` | TheLastBen/Papercut_SDXL | `papercut` | 1024×1024 | Paper-cut / kirigami craft art style |
| `sdxl_hypersd_lora.json` | ByteDance/Hyper-SD | — | 1024×1024 | Hyper-SD 8-step CFG-preserved acceleration LoRA; `lora_scale: 0.125`, `guidance_scale: 5–8` |

### FLUX (`pipeline_type: "flux"`)

> Gated model — requires a HuggingFace token stored in `.hf_token`.

| File | LoRA / adapter | Trigger word | Size | Description |
|---|---|---|---|---|
| `flux1_schnell.json` | — | — | 1024×1024 | FLUX.1-schnell — 12B rectified-flow DiT, 4-step distillation; `guidance_scale: 0.0` |
| `flux_cute_comic_lora.json` | fffiloni/cute-comic-800 | `in the style of TOK` | 1024×1024 | Charming flat illustration / comic card style |
| `flux_miniature_people_lora.json` | iliketoasters/miniature-people | `miniature people` | 1024×1024 | Photorealistic tiny figures in real-world scenes |

### Z-Image-Turbo (`pipeline_type: "zimage"`) — archived

> `guidance_scale` must be `0.0`. Recommended steps: 8–16. Configs moved to `nfsw/`.

### Archived configs

Configs in sub-folders are excluded from the web dashboard dropdown:

| Folder | Reason |
|---|---|
| `nfsw/` | NSFW / adult-content models — use at your own risk |
| `high_memory/` | Models that exceed available RAM / VRAM on the target machine |

#### `high_memory/`

| File | LoRA / adapter | Trigger word | Size | Description |
|---|---|---|---|---|
| `qwen_edit_pixel_spritesheet_lora.json` | svntax-dev/pixel_spritesheet_… | — | 768×768 | Qwen-Image-Edit + pixel spritesheet LoRA — 32×48 RPG walk/combat spritesheets |

#### `nfsw/`

> ⚠️ Adult content — excluded from the web dashboard dropdown.

| File | LoRA / adapter | Trigger word | Size | Description |
|---|---|---|---|---|
| `zimage_smnth_nsfw_lora.json` | Kakelaka/Smnth_v1_NSFW1 | `Smnth_v1` | 1024×1024 | Z-Image-Turbo + anatomical detail / NSFW character LoRA |
| `qwen_hmfemme_lora.json` | burnerbaby/hmfemme-realistic-1girl-lora-for-qwen | `HMFemme, an amateur photo…` | 1328×1328 | Qwen-Image + candid-style realistic female photography LoRA |
