# Local Text-to-Image Inference

Run text-to-image AI models entirely on your local machine — no cloud, no API key required (except for a few gated HuggingFace models). Images are generated via a command-line interface for quick ad-hoc use or queued through a background worker and web dashboard for unattended batch generation.

A growing collection of ready-to-use JSON config files covers a wide range of models, styles, LoRA weights, and adapters. Adding a new model is as simple as dropping a new `.json` file into `configs/`.

**Supported hardware:** Apple Silicon (MPS), NVIDIA GPU (CUDA), CPU fallback.

---

## Quick start

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Store your HuggingFace token for gated models (FLUX, SD3)
echo "hf_..." > .hf_token

# 3. Generate an image
./run.sh "a misty forest at dawn"
```

---

## CLI — ad-hoc generation

`run.sh` wraps `generate.py` and handles virtual-environment activation automatically.

```bash
# Use the default SD 1.5 config
./run.sh "a misty forest at dawn"

# Choose a specific config
./run.sh -c configs/sdxl_graffiti_lora.json "graarg graffiti mural of a dragon"

# Override config parameters on the fly
./run.sh -c configs/sd15_default.json --steps 50 --guidance-scale 8 "a cat"

# Save to a specific file
./run.sh -c configs/flux1_schnell.json -o outputs/my_image.png "neon city"

# Add to batch queue instead of generating immediately
./run.sh --queue -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"
```

### CLI flags

| Flag | Short | Description |
|---|---|---|
| `PROMPT` | | Text prompt describing the image **(positional, required)** |
| `--config FILE` | `-c` | Path to a JSON config file. Defaults to `configs/sd15_default.json`. |
| `--negative-prompt TEXT` | `-n` | Negative prompt — what to avoid in the image. |
| `--output FILE` | `-o` | Output PNG path. Default: `outputs/YYYYMMDD_HHMMSS.png`. |
| `--model-id REPO_ID` | | Override `model_id` from config. |
| `--adapter-id REPO_ID` | | Override `adapter_id` from config. |
| `--lora-id REPO_ID` | | Override `lora_id` from config. |
| `--lora-scale FLOAT` | | Override `lora_scale` from config. |
| `--steps N` | | Override `num_inference_steps` from config. |
| `--guidance-scale FLOAT` | | Override `guidance_scale` from config. |
| `--queue` | | Add job to batch queue instead of generating immediately. |
| `--offline` | | Skip HuggingFace update checks. Sets `HF_HUB_OFFLINE=1` so no network requests are made. Faster startup when all models are already downloaded. Fails if a model is not fully cached locally. LoRA configs must have `weight_name` set when used with `--offline`. |
| `--help` | `-h` | Show help and exit. |

CLI flags > config file > built-in defaults.

---

## CLI — queuing a single job

Pass `--queue` to `run.sh` to add a job to the batch queue instead of generating immediately. All other flags work exactly the same way.

If the worker is not yet running, `run.sh` starts it automatically in the background. Worker output is logged to `batch/worker.log`.

```bash
# Queue a job — worker is started automatically if not already running
./run.sh --queue "a neon city"
./run.sh --queue -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"
./run.sh --queue -c configs/flux1_schnell.json --steps 4 "a futuristic skyline"
```

`run.sh` prints the assigned job ID and current queue stats after adding the job. Alternatively, you can enqueue jobs directly via `batch.enqueue` (same flags, no venv handling):

```bash
python -m batch.enqueue "a neon city" -c configs/sdxl_graffiti_lora.json
```

---

## Batch server — unattended generation

For generating multiple images unattended, a background worker and a web dashboard are provided.

### Start

```bash
# Start worker + web server together (recommended)
./run_batch_server.sh              # → http://localhost:8000
./run_batch_server.sh --offline    # skip HuggingFace update checks (models must be cached)
PORT=9000 ./run_batch_server.sh    # custom port

# Or start them separately
python -m batch.worker &           # background worker
python -m batch.server             # web server (default port: 8000)
```

### Web dashboard (`http://localhost:8000`)

- **Stats bar** — live counts of pending / running / done / failed jobs
- **New Job form** — prompt, config dropdown (auto-populated from `configs/`), optional parameter overrides
- **Queue list** — status badges, auto-refresh every 3 s, expandable job details
- **Image preview** — thumbnail shown inline for finished jobs
- **Actions** — delete pending jobs, retry failed/done jobs, clear all finished jobs

### Worker flags

```bash
python -m batch.worker [--poll SECONDS] [--once]
```

| Flag | Default | Description |
|---|---|---|
| `--poll N` | `5` | Seconds between queue checks when idle |
| `--once` | off | Process exactly one pending job then exit (useful for cron) |

### Job statuses

| Status | Meaning |
|---|---|
| `pending` | Waiting in queue |
| `running` | Currently being processed |
| `done` | Finished — `result_path` points to the output file |
| `failed` | Error — `error` field contains the traceback |

### REST API

The web server exposes a JSON API at `http://localhost:8000/api/`:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/jobs` | List all jobs (oldest first) |
| `POST` | `/api/jobs` | Enqueue a new job |
| `GET` | `/api/jobs/{id}` | Get a single job |
| `DELETE` | `/api/jobs/{id}` | Delete a pending job |
| `POST` | `/api/jobs/{id}/retry` | Re-queue a failed or done job |
| `POST` | `/api/clear-finished` | Remove all done/failed jobs |
| `GET` | `/api/stats` | Job counts per status |
| `GET` | `/api/configs` | List available config files (used by the web UI dropdown) |
| `GET` | `/outputs/{filename}` | Serve a generated image |

**POST `/api/jobs` request body:**

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

---

## Configs

For a full list of available configs and the complete config file reference, see **[configs/CONFIGS.md](configs/CONFIGS.md)**.

**Supported backends:**

| `pipeline_type` | Models | RAM (approx.) |
|---|---|---|
| `sd` | Stable Diffusion 1.5, 2.1, SD3-Medium | ~4 GB (SD 1.5) |
| `sdxl` | Stable Diffusion XL, SDXL-Turbo, SDXL LoRAs | ~10 GB |
| `sd3` | Stable Diffusion 3 | ~10 GB |
| `flux` | FLUX.1-schnell, FLUX.1-dev | ~16 GB with offload |
| `zimage` | Z-Image-Turbo + LoRAs | ~16 GB |
| `qwen` | Qwen-Image | ~16 GB |

---

## Project structure

```
inference_test/
├── run.sh                  # Entry point: generate a single image via CLI
├── run_batch_server.sh     # Entry point: start worker + web server
├── generate.py             # Core generation logic — loads config, runs pipeline
├── cli.py                  # Argument parsing and config merging
├── pipeline_config.py      # Typed config container (PipelineConfig)
│
├── pipelines/              # Backend implementations
│   ├── __init__.py         # Pipeline registry + factory (create_pipeline)
│   ├── base.py             # Abstract BasePipeline class
│   ├── sd_pipeline.py      # SD 1.5 / SDXL / SD3 backend
│   ├── flux_pipeline.py    # FLUX.1-schnell / dev backend
│   ├── zimage_pipeline.py  # Z-Image-Turbo backend
│   └── qwen_pipeline.py    # Qwen-Image backend
│
├── batch/                  # Batch queue system
│   ├── queue.py            # Persistent FIFO queue backed by queue.jsonl
│   ├── worker.py           # Background worker — polls queue, runs jobs
│   ├── enqueue.py          # CLI tool to add jobs to the queue
│   └── server.py           # FastAPI web server — REST API + browser dashboard
│
├── configs/                # JSON config files, one per model/LoRA combination
│   └── CONFIGS.md          # Config reference and full list of available configs
│
├── outputs/                # Generated images (auto-created)
├── models/                 # Downloaded model weights cache (auto-created)
└── requirements.txt        # Python dependencies
```

### How it works

1. **Config resolution** — `cli.py` merges the JSON config file with any CLI flag overrides into a `PipelineConfig` object.
2. **Pipeline selection** — `pipelines/__init__.py` reads `pipeline_type` from the config and lazily imports the matching backend class from `_REGISTRY`.
3. **Model loading** — The backend downloads and caches model weights from HuggingFace on first use (stored in `models/`). LoRA weights are loaded and fused into the base model in memory.
4. **Inference** — `generate_image()` in `generate.py` calls `pipeline.generate(prompt, negative_prompt)` and saves the result as a PNG.
5. **Batch mode** — `batch/worker.py` polls `queue.jsonl` for pending jobs and calls the same `generate_image()` function. The pipeline instance is cached between consecutive jobs that share the same config, avoiding redundant model reloads.

### Adding a new backend

1. Create `pipelines/my_pipeline.py` with a class inheriting `BasePipeline`.
2. Implement `generate(prompt, negative_prompt) -> PIL.Image`.
3. Add an entry to `_REGISTRY` in `pipelines/__init__.py`.
4. Create a config file in `configs/` with `"pipeline_type": "my_type"`.

---

## Notes

- Model weights are downloaded automatically on first run and cached in `models/`.
- On Apple Silicon the MPS backend is selected automatically; CUDA is used on NVIDIA GPUs.
- Output filenames default to `YYYYMMDD_HHMMSS.png` unless `--output` is specified.
- **FLUX / SD3:** Gated models on HuggingFace — accept the license on the model page and store your token in `.hf_token`.
- **Z-Image-Turbo:** `guidance_scale` must be `0.0` — CFG is baked into the distillation. Recommended steps: 8–16.
- **SDXL on Mac:** Enable `"sequential_cpu_offload": true` in the config to avoid out-of-memory errors on 16 GB unified memory machines.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

> **Note:** The model weights used by this project have their own licenses (e.g. FLUX.1-dev is non-commercial). The MIT license covers only the code in this repository, not the model weights themselves.
