# Text-to-Image Local Inference

Run Hugging Face text-to-image models locally with a flexible CLI and JSON config system.

## Supported backends

| `pipeline_type` | Models | Notes |
|---|---|---|
| `sd` (default) | Stable Diffusion 1.5 | ~4 GB RAM |
| `sdxl` | Stable Diffusion XL, SDXL LoRAs | ~10 GB RAM |
| `anima` | Anima, AnimaYume (Cosmos-Predict2) | Requires `sd-cli` binary |

## Prerequisites

- Python 3.9+
- Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU
- **For the `anima` backend only:** `sd` binary from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

Download the pre-built macOS ARM64 release (~20 MB):
```bash
# 1. Go to https://github.com/leejet/stable-diffusion.cpp/releases/latest
# 2. Download the file ending in -bin-Darwin-macOS-*-arm64.zip
# 3. Install:
unzip sd-master-*-bin-Darwin-macOS-*-arm64.zip
sudo mv sd /usr/local/bin/
sudo chmod +x /usr/local/bin/sd
```

## Setup

```bash
# Create & activate virtual environment + install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optionally store your Hugging Face token (required for gated models)
echo "hf_..." > .hf_token
```

Or use the provided script:

```bash
./run.sh --help
```

## Usage

```bash
# SD 1.5 (default config)
./run.sh "a sunset over mountains"

# SDXL + graffiti LoRA
./run.sh -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"

# Anima / AnimaYume  (requires sd-cli)
./run.sh -c configs/animayume.json "1girl, anime, cherry blossoms"

# Override any config option on the fly
./run.sh -c configs/sd15_default.json --steps 50 --guidance-scale 8 "a cat"
```

### CLI flags

| Flag | Short | Description |
|---|---|---|
| `PROMPT` | | Text prompt describing the image to generate **(positional, required)** |
| `--config FILE` | `-c` | Path to a JSON config file. Defaults to `configs/sd15_default.json`. |
| `--negative-prompt TEXT` | `-n` | Negative prompt — what to avoid. Default: empty. |
| `--output FILE` | `-o` | Output PNG path. Default: `outputs/YYYYMMDD_HHMMSS.png`. |
| `--model-id REPO_ID` | | Override `model_id` from config. |
| `--adapter-id REPO_ID` | | Override `adapter_id` from config. |
| `--lora-id REPO_ID` | | Override `lora_id` from config. |
| `--lora-scale FLOAT` | | Override `lora_scale` from config. |
| `--steps N` | | Override `num_inference_steps` from config. |
| `--guidance-scale FLOAT` | | Override `guidance_scale` from config. |
| `--help` | `-h` | Show help and exit. |

CLI flags take precedence over the config file; the config file takes precedence over built-in defaults.

## Configuration files (`configs/`)

| File | Backend | LoRA | Trigger word | Size | Description |
|---|---|---|---|---|---|
| `sd15_default.json` | `sd` | — | — | 512×512 | Stable Diffusion 1.5, no LoRA |
| `sdxl_graffiti_lora.json` | `sdxl` | linoyts/lora-xl-graffiti-… | `graarg graffiti` | 1024×1024 | Graffiti lettering / mural style |
| `sdxl_littletinies_lora.json` | `sdxl` | alvdansen/littletinies | — | 1024×1024 | Soft hand-drawn cartoon style |
| `sdxl_ikea_lora.json` | `sdxl` | ostris/ikea-instructions-lora-sdxl | — | 1024×1024 | Flat line-art assembly-manual style |
| `sdxl_bandw_manga_lora.json` | `sdxl` | alvdansen/BandW-Manga | — | 1024×1024 | Bold monochrome line-art portrait style |
| `sdxl_storyboard_sketch_lora.json` | `sdxl` | blink7630/storyboard-sketch | `storyboard sketch of` | 1024×1024 | Grayscale film/TV storyboard style |
| `sdxl_pokemon_sprite_lora.json` | `sdxl` | sWizad/pokemon-trainer-sprite-pixelart | — | 768×768 | Pixel-art trainer sprite style |
| `animayume.json` | `anima` | — | — | 1024×1024 | Anime-style Cosmos-Predict2 fine-tune |

### Available config keys

| Key | Default | Description |
|---|---|---|
| `pipeline_type` | **(required)** | `"sd"` / `"sdxl"` / `"anima"` — must be set in every config |
| `description` | `null` | Human-readable label shown at startup |
| `model_id` | SD 1.5 repo | HF repo ID or local path of the base model |
| `adapter_id` | `null` | HF repo ID / local path for an adapter (ControlNet, refiner, …) |
| `lora_id` | `null` | HF repo ID / local path for LoRA weights |
| `lora_scale` | `0.9` | LoRA blending strength (0.0–1.0) |
| `trigger_word` | `null` | Token required by the LoRA. If missing from the prompt it is prepended automatically with a warning. |
| `num_inference_steps` | `30` | Denoising steps — more = better quality, slower |
| `guidance_scale` | `7.5` | CFG scale — how strongly the model follows the prompt |
| `width` / `height` | `1024` | Output resolution in pixels |
| `sequential_cpu_offload` | `false` | Offload sub-modules to CPU between ops; cuts peak VRAM ~50 %. Recommended for SDXL on ≤16 GB RAM. |
| `output_dir` | `"outputs"` | Directory for generated images |
| `cache_dir` | `"models"` | Directory for downloaded model weights |

## Notes

- Model weights are downloaded automatically on first run and cached in `cache_dir`.
- On Apple Silicon the MPS backend is used automatically.
- The Anima backend downloads ~14 GB of model weights on first run (diffusion model, text encoder, VAE).
- Output filenames are auto-generated as `YYYYMMDD_HHMMSS.png` unless `--output` is specified.
