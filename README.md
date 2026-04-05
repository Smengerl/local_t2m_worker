# Text-to-Image Local Inference

Run Hugging Face text-to-image models locally with a flexible CLI and JSON config system.

## Supported backends

| `pipeline_type` | Models | Notes |
|---|---|---|
| `sd` (default) | Stable Diffusion 1.5 | ~4 GB RAM |
| `sdxl` | Stable Diffusion XL, SDXL LoRAs | ~10 GB RAM |
| `anima` | Anima, AnimaYume (Cosmos-Predict2) | Requires `sd-cli` binary |
| `zimage` | Z-Image-Turbo (Tongyi-MAI) + LoRAs | ~16 GB RAM; `guidance_scale` must be `0.0` |

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
| `sd15_inkpunk_lora.json` | `sd` | — | `nvinkpunk` | 512×512 | Ink/punk illustration style (Gorillaz / FLCL aesthetic) |
| `sd15_pixel_art_lora.json` | `sd` | SedatAl/pixel-art-LoRa | — | 512×512 | Pixel-art / 16-bit retro style (trained on diffusiondb-pixelart) |
| `sdxl_graffiti_lora.json` | `sdxl` | linoyts/lora-xl-graffiti-… | `graarg graffiti` | 1024×1024 | Graffiti lettering / mural style |
| `sdxl_littletinies_lora.json` | `sdxl` | alvdansen/littletinies | — | 1024×1024 | Soft hand-drawn cartoon style |
| `sdxl_ikea_lora.json` | `sdxl` | ostris/ikea-instructions-lora-sdxl | — | 1024×1024 | Flat line-art assembly-manual style |
| `sdxl_bandw_manga_lora.json` | `sdxl` | alvdansen/BandW-Manga | — | 1024×1024 | Bold monochrome line-art portrait style |
| `sdxl_storyboard_sketch_lora.json` | `sdxl` | blink7630/storyboard-sketch | `storyboard sketch of` | 1024×1024 | Grayscale film/TV storyboard style |
| `sdxl_pokemon_sprite_lora.json` | `sdxl` | sWizad/pokemon-trainer-sprite-pixelart | — | 768×768 | Pixel-art trainer sprite style |
| `animayume.json` | `anima` | — | — | 1024×1024 | Anime-style Cosmos-Predict2 fine-tune |
| `zimage_smnth_nsfw_lora.json` | `zimage` | Kakelaka/Smnth_v1_NSFW1 | `Smnth_v1` | 1024×1024 | Z-Image-Turbo + anatomical detail / NSFW character LoRA |
| `zimage_hmfemme_lora.json` | `zimage` | burnerbaby/hmfemme-realistic-1girl-lora-for-qwen | `HMFemme, an amateur photo…` | 1024×1024 | Z-Image-Turbo + candid-style realistic female photography LoRA |
| `zimage_pornmaster_lora.json` | `zimage` | RomixERR/Pornmaster_v1-Z-Images-Turbo | `pronmstr` | 1024×1024 | Z-Image-Turbo + NSFW realism LoRA (WIP, unpredictable results) |

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

## Batch Queue System

For generating multiple images unattended, there is a built-in batch queue backed by a persistent JSONL file.
Jobs are processed sequentially in FIFO order. The pipeline is cached between jobs with the same config, so
consecutive jobs with the same model don't reload the weights.

### Components

| File | Role |
|---|---|
| `batch/queue.py` | Persistent FIFO queue (`queue.jsonl`) with filelock |
| `batch/worker.py` | Background worker — polls queue, runs jobs, caches pipeline |
| `batch/enqueue.py` | CLI tool to add jobs |
| `batch/server.py` | FastAPI app — REST API + browser dashboard |
| `run_batch_server.sh` | Convenience launcher (starts worker + web server together) |

### Quick start

```bash
# Start worker + web server in one command
./run_batch_server.sh              # → http://localhost:8000
PORT=9000 ./run_batch_server.sh    # custom port

# Or start them separately
python -m batch.worker &           # worker in background
python -m batch.server             # web server (default: localhost:8000)
```

### Web dashboard (`http://localhost:8000`)

- **Stats bar** — live counts of pending / running / done / failed jobs
- **New Job form** — prompt, config dropdown, optional overrides (steps, guidance scale, LoRA scale, …)
- **Queue list** — status badges, auto-refresh every 3 s, click to expand details
- **Image preview** — thumbnail shown inline for finished jobs
- **Actions** — delete pending jobs, retry failed/done jobs, clear all finished

### Enqueueing jobs via CLI

```bash
# Same flags as generate.py / run.sh
python -m batch.enqueue "a neon city" -c configs/sdxl_graffiti_lora.json
python -m batch.enqueue --steps 40 --guidance-scale 8 "a cat in space"
```

Prints the job ID and current queue stats after adding the job.

### REST API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/jobs` | List all jobs (oldest first) |
| `POST` | `/api/jobs` | Enqueue a new job (JSON body) |
| `GET` | `/api/jobs/{id}` | Get a single job |
| `DELETE` | `/api/jobs/{id}` | Delete a pending job |
| `POST` | `/api/jobs/{id}/retry` | Re-queue a failed or done job |
| `POST` | `/api/clear-finished` | Remove all done/failed jobs |
| `GET` | `/api/stats` | Counts per status |
| `GET` | `/api/configs` | List all available config files (used by web UI dropdown) |
| `GET` | `/outputs/{filename}` | Serve a generated image |

**POST `/api/jobs` body:**

```json
{
  "prompt": "a neon city",
  "config": "configs/sdxl_graffiti_lora.json",
  "negative_prompt": "",
  "steps": null,
  "guidance_scale": null,
  "lora_scale": null,
  "model_id": null,
  "lora_id": null,
  "adapter_id": null,
  "output": null
}
```

### Worker flags

```
python -m batch.worker [--poll SECONDS] [--once]
```

| Flag | Default | Description |
|---|---|---|
| `--poll N` | `5` | Seconds to wait between queue checks when idle |
| `--once` | off | Process one pending job then exit (useful for cron/testing) |

### Job statuses

| Status | Meaning |
|---|---|
| `pending` | Waiting in queue |
| `running` | Currently being processed |
| `done` | Finished — `result_path` contains the output file path |
| `failed` | Error — `error` field contains the traceback |

## Notes

- Model weights are downloaded automatically on first run and cached in `cache_dir`.
- On Apple Silicon the MPS backend is used automatically.
- The Anima backend downloads ~14 GB of model weights on first run (diffusion model, text encoder, VAE).
- Output filenames are auto-generated as `YYYYMMDD_HHMMSS.png` unless `--output` is specified.
- **Z-Image-Turbo (`zimage`):** Uses `ZImagePipeline` from diffusers (requires diffusers ≥ 0.33 or installed from source). `guidance_scale` **must** be `0.0` — the Turbo variant has CFG baked into distillation. Recommended steps: 8–16. dtype: bfloat16 (auto-selected on CUDA/MPS).
