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
./scripts/run.sh "a misty forest at dawn"
```

---

## CLI — ad-hoc generation

`scripts/run.sh` wraps `generate.py` and handles virtual-environment activation automatically.

```bash
# Use the default SD 1.5 config
./scripts/run.sh "a misty forest at dawn"

# Choose a specific config
./scripts/run.sh -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"

# Override config parameters on the fly
./scripts/run.sh -c configs/sd15_default.json --steps 50 --guidance-scale 8 "a cat"

# Save to a specific file
./scripts/run.sh -c configs/flux1_schnell.json -o outputs/my_image.png "neon city"

# Add to batch queue instead of generating immediately
./scripts/run.sh --queue -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"
```

For the full list of flags see **[scripts/README.md](scripts/README.md#runsh)**.

---

## CLI — queuing a single job

Pass `--queue` to `scripts/run.sh` to add a job to the batch queue instead of generating immediately. All other flags work exactly the same way.

If the worker is not yet running, `scripts/run.sh` starts it automatically in the background. Worker output is logged to `batch/worker.log`.

```bash
# Queue a job — worker is started automatically if not already running
./scripts/run.sh --queue "a neon city"
./scripts/run.sh --queue -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"
./scripts/run.sh --queue -c configs/flux1_schnell.json --steps 4 "a futuristic skyline"
```

`scripts/run.sh` prints the assigned job ID and current queue stats after adding the job. Alternatively, you can enqueue jobs directly via `batch.enqueue` (same flags, no venv handling):

```bash
python -m batch.enqueue "a neon city" -c configs/sdxl_graffiti_lora.json
```

---

## Batch server — unattended generation

For generating multiple images unattended, a background worker and a web dashboard are provided.

### Start

```bash
# Start worker + web server together (recommended)
./scripts/run_batch_server.sh              # → http://localhost:8000
./scripts/run_batch_server.sh --offline    # skip HuggingFace update checks (models must be cached)
PORT=9000 ./scripts/run_batch_server.sh    # custom port

# Or start them separately
python -m batch.worker &           # background worker
python -m batch.server             # web server (default port: 8000)
```

### Health check

```bash
./scripts/health_check.sh           # default PORT=8000
PORT=9000 ./scripts/health_check.sh # custom port
```

All status data is fetched from a single `GET /api/health` call. The script shows worker task liveness, current job, pipeline cache state, queue counters, and network rates. It refreshes every ~5 seconds until Ctrl-C.

See **[scripts/README.md](scripts/README.md#health_checksh)** for details on what each status line checks.

### Runtime behaviour

#### Model switching and pipeline cache

Only one model is held in memory at a time. When consecutive jobs use the same config the pipeline is reused — no reload needed. When a job uses a different config the previous model is evicted from memory before the new one is loaded. **Model loading takes several minutes for large models (FLUX, SDXL).** Plan batch jobs to minimise config switches.

#### Shutdown timeout

When the server is stopped (Ctrl-C or SIGTERM) the worker is given **10 seconds** to finish the current denoising step and save the result. After that the shutdown proceeds regardless. The job is marked as `failed` if it could not be completed in time.

#### Live generation logs in the web UI

Progress bars from `tqdm` and HuggingFace download progress are captured and streamed into each job's log. Expand a job in the dashboard or poll `GET /api/jobs/{id}` to see live output.

#### Offline mode

Pass `--offline` to `run_batch_server.sh` (or `run.sh`) to skip all HuggingFace network calls. Diffusers normally performs a lightweight HEAD request on each model load to check for updates — `--offline` suppresses this. The model must be fully cached locally (use `scripts/preload_model.sh` first).

```bash
./scripts/run_batch_server.sh --offline
./scripts/run.sh --offline "a misty forest"
```

### Web dashboard (`http://localhost:8000`)

- **Stats bar** — live counts of pending / running / done / failed jobs
- **New Job form** — prompt, config dropdown (auto-populated from `configs/`), optional parameter overrides
- **Queue list** — status badges, auto-refresh every 3 s, expandable job details
- **Image preview** — thumbnail shown inline for finished jobs
- **Actions** — delete pending jobs, retry failed/done jobs, clear all finished jobs

### Worker flags

```bash
python -m batch.worker [--keep-alive]
```

| Flag | Default | Description |
|---|---|---|
| `--keep-alive` | off | Stay alive after the queue is empty and wait for new jobs. Without this flag the worker exits as soon as all current pending jobs are done (one-shot mode, useful for cron). |

> **Note:** When running via `run_batch_server.sh` the worker is embedded inside the FastAPI server process as an asyncio task — the `--keep-alive` behaviour is always active and the standalone worker binary is not used.

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
| `flux2_klein` | FLUX.2 [klein] 4B | ~13 GB with offload |
| `zimage` | Z-Image-Turbo + LoRAs | ~16 GB |
| `qwen` | Qwen-Image | ~16 GB |

---

## Project structure

```
inference_test/
├── scripts/                # Entry points and utilities (see scripts/README.md)
│   ├── run.sh              # Generate a single image via CLI
│   ├── run_batch_server.sh # Start worker + web server together
│   ├── health_check.sh     # Live status check (worker, server, download rate)
│   └── preload_model.sh    # Pre-download model weights (resume-safe)
│
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
├── examples/               # Showcase scripts — enqueue one job per config (see examples/README.md)
│
├── scripts/                # Entry points and utilities — see scripts/README.md
│
├── outputs/                # Generated images (auto-created)
├── models/                 # Downloaded model weights cache (auto-created)
└── requirements.txt        # Python dependencies
```

### How it works

1. **Config resolution** — `cli.py` merges the JSON config file with any CLI flag overrides into a `PipelineConfig` object.
2. **Pipeline selection** — `pipelines/__init__.py` reads `pipeline_type` from the config and lazily imports the matching backend class from `_REGISTRY`. Heavy dependencies (torch, diffusers) are only imported when a pipeline is actually created, keeping server startup fast.
3. **Model loading** — The backend downloads and caches model weights from HuggingFace on first use (stored in `models/`). LoRA weights are loaded and fused into the base model in memory.
4. **Inference** — `generate_image()` in `generate.py` calls `pipeline.generate(prompt, negative_prompt)` and saves the result as a PNG. This is the **single authorised entry point** for all generation — both CLI and batch worker call only this function.
5. **Batch mode** — `batch/worker.py` runs as an asyncio coroutine (inside the FastAPI server process or as a standalone CLI worker). It dequeues pending jobs from `queue.jsonl` and calls `generate_image()` in a thread pool so the event loop stays free. The pipeline instance is cached between consecutive jobs that share the same config, avoiding redundant model reloads. Only one model is held in memory at a time.
6. **Web server** — `batch/server.py` embeds the worker as an in-process asyncio task and exposes the queue and results via a REST API and browser dashboard.

### Adding a new backend

1. Create `pipelines/my_pipeline.py` with a class inheriting `BasePipeline`.
2. Implement `generate(prompt, negative_prompt) -> PIL.Image`.
3. Add an entry to `_REGISTRY` in `pipelines/__init__.py`.
4. Create a config file in `configs/` with `"pipeline_type": "my_type"`.

### Architecture layers

The codebase is organised in three layers that only depend downward:

```text
Layer 3 — batch/server.py + batch/static/   Web server + dashboard
               extends ↑
Layer 2 — batch/worker.py + batch/queue.py  CLI worker + queue
               extends ↑
Layer 1 — generate.py + cli.py + pipelines/ Core CLI generation
```

`generate.py` has no knowledge of queues or HTTP. `batch/worker.py` has no knowledge of FastAPI. New features must be added at the correct layer — extending downward dependencies is not permitted.

---

## Pre-downloading models (resume-safe)

Use `scripts/preload_model.sh` to pre-download all weights for a config before starting generation. Already-cached blobs are skipped, incomplete transfers are resumed.

```bash
./scripts/preload_model.sh -c configs/flux_schnell.json
./scripts/preload_model.sh -c configs/flux_schnell.json --dry-run
```

See **[scripts/README.md](scripts/README.md#preload_modelsh)** for full usage, all flags, and details on GGUF-specific behaviour.

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
