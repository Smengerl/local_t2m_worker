# Text-to-Image Local Inference Project

Python project for running Hugging Face text-to-image models (Stable Diffusion) locally using the `diffusers` library.

## Stack
- Python 3.10+
- `diffusers` (Hugging Face)
- `transformers`
- `torch` (PyTorch)
- `Pillow`
- `accelerate`
- `peft` (LoRA support)
- `huggingface_hub`

## Project Structure

```
inference_test/
├── generate.py          – Entry point: parses args, loads config, runs pipeline
├── cli.py               – All CLI argument parsing, config loading, trigger-word logic
├── run.sh               – Convenience wrapper: activates venv, injects default config
├── requirements.txt     – Python dependencies
├── configs/             – JSON config files (one per model/style preset)
├── pipelines/
│   ├── __init__.py      – create_pipeline(cfg) factory + _REGISTRY
│   ├── base.py          – BasePipeline(ABC) with abstract generate()
│   ├── sd_pipeline.py   – StableDiffusionBackend (SD 1.5 + SDXL via diffusers)
│   └── anima_pipeline.py – AnimaPipeline (Cosmos-Predict2 via sd-cli binary)
├── outputs/             – Generated images (auto-timestamped)
└── models/              – Downloaded model weights cache
```

## Pipeline Architecture

Every config must set `pipeline_type`. The factory in `pipelines/__init__.py` maps it to a class:

| `pipeline_type` | Class | Library |
|---|---|---|
| `sd` | `StableDiffusionBackend` | diffusers |
| `sdxl` | `StableDiffusionBackend` | diffusers |
| `anima` | `AnimaPipeline` | sd-cli subprocess |

All pipeline classes inherit from `BasePipeline(ABC)` and implement `generate(prompt, negative_prompt) -> Image`.

## Config File Parameters

All JSON config files live in `configs/`. Keys starting with `_` are treated as comments and ignored at runtime.

### Complete list of recognised keys

| Key | Type | Default | Description |
|---|---|---|---|
| `pipeline_type` | string | **(required)** | `"sd"` / `"sdxl"` / `"anima"` |
| `description` | string | `null` | Human-readable description shown at startup |
| `model_id` | string | `"stable-diffusion-v1-5/stable-diffusion-v1-5"` | HF repo ID or local path of the base model |
| `adapter_id` | string\|null | `null` | HF repo ID / local path for an adapter (ControlNet, refiner, …) |
| `lora_id` | string\|null | `null` | HF repo ID / local path for LoRA weights |
| `lora_scale` | float | `0.9` | LoRA blending strength (0.0–1.0) |
| `trigger_word` | string\|null | `null` | Token required by the LoRA/model. If absent from the prompt it is prepended automatically with a warning. |
| `num_inference_steps` | int | `30` | Denoising steps — more = better quality, slower |
| `guidance_scale` | float | `7.5` | CFG scale — how strongly the model follows the prompt |
| `width` | int | `1024` | Output image width in pixels |
| `height` | int | `1024` | Output image height in pixels |
| `sequential_cpu_offload` | bool | `false` | Offload model sub-modules to CPU between ops; cuts peak VRAM ~50 % at cost of speed. Recommended for SDXL on ≤16 GB unified memory. Ignored by the `anima` pipeline. |
| `output_dir` | string | `"outputs"` | Directory for generated images |
| `cache_dir` | string | `"models"` | Directory for downloaded model weights |

> **Important:** `pipeline_type` is **mandatory** — omitting it raises a `KeyError` at startup.  
> `adapter_id` is wired up in the config schema but not yet consumed by any pipeline class (reserved for future ControlNet / refiner support).

### Trigger-word behaviour

When `trigger_word` is set in a config, `build_config()` in `cli.py` checks (case-insensitively) whether the token appears in the user's prompt. If it is missing, the token is **prepended automatically** and a `⚠️` warning is printed. The modified prompt is stored in `cfg["_effective_prompt"]` and passed to the pipeline instead of the raw `args.prompt`.

## Available Config Presets

| File | `pipeline_type` | Base model | LoRA | Trigger word | Size |
|---|---|---|---|---|---|
| `sd15_default.json` | `sd` | SD 1.5 | — | — | 512×512 |
| `sdxl_graffiti_lora.json` | `sdxl` | SDXL 1.0 | linoyts/lora-xl-graffiti-… | `graarg graffiti` | 1024×1024 |
| `sdxl_littletinies_lora.json` | `sdxl` | SDXL 1.0 | alvdansen/littletinies | — | 1024×1024 |
| `sdxl_ikea_lora.json` | `sdxl` | SDXL 1.0 | ostris/ikea-instructions-lora-sdxl | — | 1024×1024 |
| `sdxl_bandw_manga_lora.json` | `sdxl` | SDXL 1.0 | alvdansen/BandW-Manga | — | 1024×1024 |
| `sdxl_storyboard_sketch_lora.json` | `sdxl` | SDXL 1.0 | blink7630/storyboard-sketch | `storyboard sketch of` | 1024×1024 |
| `sdxl_pokemon_sprite_lora.json` | `sdxl` | SDXL 1.0 | sWizad/pokemon-trainer-sprite-pixelart | — | 768×768 |
| `animayume.json` | `anima` | duongve/AnimaYume | — | — | 1024×1024 |

## CLI Reference (`generate.py` / `run.sh`)

`run.sh` is a thin wrapper around `python generate.py`. All flags are identical.

```
Usage: ./run.sh [OPTIONS] "PROMPT"
   or: python generate.py [OPTIONS] "PROMPT"
```

### Positional argument

| Argument | Description |
|---|---|
| `PROMPT` | Text prompt describing the image to generate (required) |

### Options

| Flag | Short | Metavar | Description |
|---|---|---|---|
| `--config` | `-c` | `FILE` | Path to a JSON config file. Defaults to `configs/sd15_default.json`. |
| `--negative-prompt` | `-n` | `TEXT` | Negative prompt — what to avoid. Default: empty string. |
| `--output` | `-o` | `FILE` | Output PNG path. Default: `outputs/YYYYMMDD_HHMMSS.png`. |
| `--model-id` | | `REPO_ID` | Override `model_id` from config. |
| `--adapter-id` | | `REPO_ID` | Override `adapter_id` from config. |
| `--lora-id` | | `REPO_ID` | Override `lora_id` from config. |
| `--lora-scale` | | `FLOAT` | Override `lora_scale` from config. |
| `--steps` | | `N` | Override `num_inference_steps` from config. |
| `--guidance-scale` | | `FLOAT` | Override `guidance_scale` from config. |
| `--help` | `-h` | | Show help and exit. |

CLI flags take precedence over the config file; the config file takes precedence over built-in defaults.

## README.md Maintenance Rules

> **For Copilot:** when editing `README.md`, always keep the following sections in sync with the actual codebase:
>
> 1. **"Configuration files" table** — must list every `.json` file in `configs/` with its pipeline type and `description` value.
> 2. **"Available config keys" table** — must match the `DEFAULTS` dict in `cli.py` exactly. Every key present in `DEFAULTS` (plus `pipeline_type` and `description`) must appear; no key that was removed from `DEFAULTS` should remain.
> 3. **CLI flags table** — must match every `parser.add_argument(...)` call in `cli.py`'s `parse_args()` function. Check flag names, short aliases, metavar strings, and descriptions.
> 4. **"Supported backends" table** — must match the `_REGISTRY` dict in `pipelines/__init__.py`.
>
> Never update the README without first reading the current `cli.py` DEFAULTS, `parse_args()`, and `configs/` directory listing.

## Notes
- Use MPS (Apple Silicon) or CUDA if available, otherwise CPU
- Model weights are cached in `cache_dir` (default: `models/` inside the project)
- The `anima` pipeline requires the `sd` binary from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — roughly 14 GB downloaded on first run
- Output filenames are auto-generated as `YYYYMMDD_HHMMSS.png` unless `--output` is specified
- Keys starting with `_` in JSON config files are treated as inline comments and ignored at runtime
