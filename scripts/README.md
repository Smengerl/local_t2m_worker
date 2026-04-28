# Scripts

All executable entry points for the project live in this directory.

| Script | Type | Purpose |
|---|---|---|
| [`run.sh`](#runsh) | Shell | Generate a single image via CLI |
| [`run_batch_server.sh`](#run_batch_serversh) | Shell | Start worker + web server together |
| [`health_check.sh`](#health_checksh) | Shell | Check server system status at a glance |
| [`preload_model.sh`](#preload_modelsh) | Shell | Pre-download model weights (resume-safe) |
| [`preload_model.py`](preload_model.py) | Python | Implementation — called by `preload_model.sh` |

All scripts are run from the **project root** or from this directory — they resolve their own paths automatically.

> **Example scripts** (showcase jobs per config) have moved to **[`examples/`](../examples/README.md)**.

---

## `run.sh`

Wraps `generate.py` for ad-hoc image generation. Handles virtual-environment activation automatically.

### Robust single-worker logic

When using `--queue`, the script ensures that only one worker process is running at any time:

- Before starting a worker, a file lock (`flock` on `/tmp/worker.lock`) is acquired. Only the process that holds the lock can start the worker.
- After acquiring the lock, the script checks again if a worker is already running (using the PID file). If so, it does not start a new worker.
- The worker process itself also checks on startup if another worker is running (by reading the PID file and checking if the process is alive). If another worker is detected, it exits immediately.

This mechanism prevents race conditions and guarantees that jobs in the queue are always processed sequentially by a single worker process.

```bash
# Basic usage
./scripts/run.sh "a misty forest at dawn"

# Choose a config
./scripts/run.sh -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"

# Override config parameters on the fly
./scripts/run.sh -c configs/sd15_default.json --steps 50 --guidance-scale 8 "a cat"

# Save to a specific file
./scripts/run.sh -c configs/flux1_schnell.json -o outputs/my_image.png "neon city"

# Add to batch queue instead of generating immediately
./scripts/run.sh --queue -c configs/sdxl_graffiti_lora.json "graffiti mural of a dragon"

# Skip HuggingFace network checks (models must be fully cached)
./scripts/run.sh --offline "a misty forest at dawn"
```

### Flags

| Flag | Short | Description |
|---|---|---|
| `PROMPT` | | Text prompt **(positional, required)** |
| `--config FILE` | `-c` | JSON config file. Default: `configs/sd15_default.json` |
| `--negative-prompt TEXT` | `-n` | Negative prompt — what to avoid |
| `--output FILE` | `-o` | Output PNG path. Default: `outputs/YYYYMMDD_HHMMSS.png` |
| `--model-id REPO_ID` | | Override `model_id` from config |
| `--lora-id REPO_ID` | | Override `lora_id` from config |
| `--lora-scale FLOAT` | | Override `lora_scale` from config |
| `--steps N` | | Override `num_inference_steps` from config |
| `--guidance-scale FLOAT` | | Override `guidance_scale` from config |
| `--queue` | | Add job to batch queue instead of generating immediately |
| `--offline` | | Set `HF_HUB_OFFLINE=1` — skip all HuggingFace network calls. Fails if model is not cached. LoRA configs must have `weight_name` set. |
| `--help` | `-h` | Show help and exit |

Priority: CLI flags > config file > built-in defaults.

---

## `run_batch_server.sh`

Starts the background worker and the web server together. The worker is killed automatically when the script exits (Ctrl-C).

```bash
# Start on default port 8000
./scripts/run_batch_server.sh

# Skip HuggingFace network checks (models must be cached)
./scripts/run_batch_server.sh --offline

# Custom port
PORT=9000 ./scripts/run_batch_server.sh
```

### Flags

| Flag | Description |
|---|---|
| `--offline` | Set `HF_HUB_OFFLINE=1` — faster startup, no network calls. Fails if a model is not fully cached. |

The web dashboard is available at `http://localhost:8000` (or the custom port) once the server starts.

---

## `health_check.sh`

Cyclic status check for the server that refreshes every ~5 seconds (Ctrl-C to quit). Useful for monitoring during a generation or download.

```bash
./scripts/health_check.sh           # default PORT=8000
PORT=9000 ./scripts/health_check.sh # custom port
```

**Check order:**

1. **Server process running** — `pgrep -f batch.server` finds the server process
2. **Server reachable** — HTTP probe on `http://localhost:PORT/` succeeds
3. **Worker/queue/model info** — Only if server is reachable: queries `/api/health` for batch worker, queue, and loaded model status
4. **Network rates** — RX rate on external interface (HF download) and loopback `lo0` (Web GUI traffic)

| Line | What is checked |
|---|---|
| **Server process running** | `pgrep -f batch.server` finds the server process |
| **Server reachable** | HTTP probe on `http://localhost:PORT/` succeeds |
| **Batch worker running** | `/api/health` reports worker alive |
| **Generation running** | `/api/health` reports at least one job with `status=running` |
| **Loaded model** | `/api/health` shows which model (if any) is in memory |
| **Queue status** | `/api/health` shows pending/running/done/failed jobs |
| **HF download** | RX rate on the external network interface (sampled over 1 s via `netstat -ib`) |
| **Web GUI traffic** | RX rate on loopback `lo0` — indicates browser activity |

> During a large model download (e.g. a GGUF file), *Generation running* will show ✖ while *HF download* shows a non-zero rate — this is expected. The job status only flips to `running` after the model is fully loaded.

---

## `preload_model.sh`

Pre-downloads all model weights required by one or more config files. Handles virtual-environment activation automatically. Already-cached blobs are skipped, partially-downloaded blobs (`.incomplete`) are resumed — nothing is re-downloaded from scratch.

```bash
# Download all weights needed by a config
./scripts/preload_model.sh -c configs/flux_schnell.json

# Download multiple configs at once
./scripts/preload_model.sh -c configs/flux_schnell.json configs/sdxl_graffiti_lora.json

# Custom cache directory (must match --cache-dir used during generation)
./scripts/preload_model.sh -c configs/flux_dev_gguf.json --cache-dir models/

# Dry-run: show exactly what would be downloaded without downloading anything
./scripts/preload_model.sh -c configs/flux_schnell.json --dry-run
```

After a successful pre-download, pass `--offline` to `run.sh` / `run_batch_server.sh` to skip all HuggingFace network checks on startup:

```bash
./scripts/run_batch_server.sh --offline
```

### Flags (preload)

| Flag | Description |
|---|---|
| `-c FILE [FILE …]` | One or more JSON config files **(required)** |
| `--cache-dir DIR` | Cache directory for weights. Default: value from config, then `~/.cache/huggingface` |
| `--token-file FILE` | Path to HF token file. Default: `.hf_token` in project root |
| `--dry-run` | Print what would be downloaded without downloading anything |
| `--help` / `-h` | Show help and exit |

### How resume works

`huggingface_hub` stores each blob as a `<sha256>` file. Incomplete transfers are kept as `<sha256>.incomplete`. On the next run:

- **Complete blobs** → skipped entirely
- **`.incomplete` blobs** → resumed from where the transfer stopped
- **Missing blobs** → downloaded fresh

### GGUF configs

For configs with a `gguf_file` field (e.g. `flux_dev_*`), the script downloads only what the pipeline actually needs:

| Source | What | Why |
|---|---|---|
| `model_id` repo | Only the specific `.gguf` file + repo metadata | The quantised transformer; other `.gguf` variants are skipped |
| `base_model_id` repo | Everything **except** `transformer/` | VAE, text encoders, tokenizer, scheduler — the full-precision transformer (~24 GB) is never downloaded |
| `lora_id` repo | Only the specific `.safetensors` file named by `weight_name` | Matches `load_lora_weights()` behaviour exactly |

> **NSFW configs** (in `configs/nsfw/`) are intentionally excluded.
