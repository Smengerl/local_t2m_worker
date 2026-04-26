# Text-to-Image Local Inference Project

Python project for running Hugging Face text-to-image models (Stable Diffusion) locally using the `diffusers` library.

> For usage, CLI flags, config keys, available presets, and batch system documentation see **README.md**.

## Architecture: Strict Layer Hierarchy (CRITICAL)

The codebase is structured in three layers. **Dependencies must only flow downward — never upward.**

```
Layer 3 ── batch/server.py  +  batch/static/  (FastAPI server + Web UI)
               │  imports from / extends Layer 2
Layer 2 ── batch/worker.py  +  batch/queue.py  +  batch/notify.py  (CLI Worker)
               │  imports from / extends Layer 1
Layer 1 ── generate.py  +  cli.py  +  pipelines/  +  pipeline_config.py  (Core CLI)
```

**Layer 1 — Core CLI** (`generate.py`, `cli.py`, `pipelines/`, `pipeline_config.py`, `config_types.py`)
: Standalone image generation. Has **no knowledge** of queues, servers, or web UIs.
: Entry point: `python generate.py` / `./run.sh`

**Layer 2 — CLI Worker** (`batch/worker.py`, `batch/queue.py`, `batch/notify.py`, `batch/api/`)
: Extends Layer 1 with a persistent FIFO queue and the async worker loop.
: Calls `generate_image()` from Layer 1. Has **no knowledge** of FastAPI or HTTP servers.
: Entry point: `python -m batch.worker`

**Layer 3 — Server + Web UI** (`batch/server.py`, `batch/static/`)
: Extends Layer 2 by embedding the worker as an in-process asyncio task and exposing everything via HTTP.
: Calls `run_worker_async()` from Layer 2. The Web UI speaks only to the REST API.
: Entry point: `python -m batch.server`

### Forbidden dependency directions

| ❌ Never do this | Why |
|---|---|
| `generate.py` imports from `batch/` | Layer 1 must not know about Layer 2 |
| `batch/worker.py` imports from `batch/server.py` | Layer 2 must not know about Layer 3 |
| `batch/queue.py` imports from FastAPI | Layer 2 must not depend on the HTTP framework |
| `pipelines/` imports from `batch/` | Core pipeline backends are Layer 1 — no queue knowledge |

### Practical implications for code changes

- Adding a new generation parameter → change Layer 1 (`config_types.py`, `pipeline_config.py`, `cli.py`, `pipelines/`). Layers 2 and 3 pick it up automatically via `PipelineConfig`.
- Adding a new worker behaviour → change Layer 2 (`batch/worker.py`). Layer 3 (`server.py`) exposes it transparently.
- Adding a new API endpoint or UI feature → change Layer 3 only. Never reach into Layer 1 to add HTTP-specific code there.
- `generate_image()` in `generate.py` is the **only authorised entry point** from Layer 2 into Layer 1. Do not call `create_pipeline()` or individual `BasePipeline` methods directly from `batch/`.

## Runtime Design Constraints (CRITICAL)

These invariants must be preserved across all code changes.

### 1. Threading model: `process_job()` is always blocking, always in a thread

`process_job()` runs PyTorch inference and **must always be called via `loop.run_in_executor()`** — never `await`ed directly and never called from the event loop thread. The contract is:

```
asyncio Event Loop (main thread)
  └─ run_worker_async()            ← async coroutine in the loop
       └─ loop.run_in_executor()   ← dispatches to ThreadPoolExecutor
            └─ process_job()       ← blocking, PyTorch inference here
```

Calling `process_job()` directly from an async context would block the entire FastAPI server.

### 2. Two-level cancellation — do not confuse them

There are two independent cancellation mechanisms:

| Level | Mechanism | Effect |
|---|---|---|
| **Job cancel** | `_cancel_event` (`threading.Event`), set by `request_cancel()` | Aborts current inference job; worker loop continues |
| **Worker shutdown** | `asyncio.CancelledError` on the task | Tears down `run_worker_async` entirely; triggers `finally` cleanup |

`_cancel_event` is checked between denoising steps in the `_progress` callback inside `process_job()`. Never use `CancelledError` to cancel a single job, and never use `_cancel_event` to shut down the worker.

### 3. Pipeline cache holds exactly one model at a time

`pipeline_cache` is a plain `dict` that **must never contain more than one entry**. Before storing a newly loaded pipeline, `generate.py` calls `pipeline_cache.clear()` to evict the previous model. Any code that adds a second entry to the cache will cause an OOM crash (each model is 5–20 GB).

The module-level `_cached_model: str | None` in `batch/worker.py` mirrors this state and is read by `/api/health`. It is updated from the cache key after every job:
```python
_cached_model = next((k[0] for k in pipeline_cache if isinstance(k, tuple) and k), None)
```
If `PipelineConfig.pipeline_cache_key()` ever changes the tuple element order, this line must be updated to match.

### 4. `_release_pipeline_cache()` must always run in the `finally` block

On Apple Silicon (MPS), PyTorch crashes the interpreter with a refcount error if pipeline objects are garbage-collected after the MPS backend has started tearing down. `_release_pipeline_cache()` prevents this by calling `gc.collect()` + `torch.mps.empty_cache()` **before** the interpreter begins to exit.

This call must remain in the `finally` block of `run_worker_async()`. Never move it into a conditional branch or remove it.

### 5. `notify.init()` is owned by the worker, not by callers

`run_worker_async(keep_alive=True)` calls `notify.init()` internally. Callers (e.g. `batch/server.py`) must **not** call `notify.init()` themselves. Likewise, `notify.reset()` is called in the `finally` block of `run_worker_async` — never elsewhere.

This keeps `batch/notify.py` state lifecycle strictly coupled to the worker task.

### 6. Queue file: full rewrite on every update — no partial writes

The JSONL queue file is protected by `filelock`. Every mutating operation (enqueue, status update, log append) calls `_write_all()`, which rewrites the **entire file**. There are no append-only or in-place mutations. Code must never write directly to `queue.jsonl` without holding the lock, and must never assume partial-write atomicity.

### 7. `_heal_stale_running_jobs()` must be called on every `GET /api/jobs`

`GET /api/jobs` calls `_heal_stale_running_jobs()` on every request. This auto-marks jobs as `failed` when their recorded `worker_pid` is no longer alive. This is the only recovery path for jobs that got stuck in `running` after a worker crash. Do not remove this call or move it behind a flag.

### 8. Pipeline backends are loaded lazily — never import them at module level

`pipelines/__init__.py` uses `importlib.import_module()` to load backend classes only when `create_pipeline()` is called. This prevents importing `torch` and `diffusers` at server startup. Never add top-level imports of pipeline backend modules in `generate.py`, `batch/`, or `batch/server.py`.

### 9. `EnqueueRequest` fields, CLI flags, and config keys form a naming triplet

Every override parameter must exist consistently in three places with matching names:

| Layer | Format | Example |
|---|---|---|
| Config JSON | `section.field` | `generation.cfg_scale` |
| CLI flag | `--section-field` | `--cfg-scale` |
| `EnqueueRequest` field | `section_field` | `cfg_scale` |

Breaking this triplet (e.g. renaming only the CLI flag) causes silent mismatches between the web UI, REST API, and config file.

## Available Skills

| Skill | Trigger | Purpose |
|---|---|---|
| `config-builder` | `#config-builder` | Create a new `configs/*.json` for any Hugging Face model — researches the model card, picks the right `pipeline_type`, fills all parameters with documented value ranges. No config list needs to be updated anywhere. |
| `config-tester` | `#config-tester` | Smoke-test an existing config — runs `generate.py`, monitors for startup errors, kills the process after the first denoising step succeeds, and auto-fixes config or pipeline issues if needed. |

### Skill suggestion rules

- If the user asks to add a model, create a config, or mentions a Hugging Face model name or URL in a way that implies they want to use it for generation, **always suggest the `#config-builder` skill** before doing any manual work:
  > 💡 For this you can use the `#config-builder` skill — it researches the model card and fills all parameters with documented ranges automatically.
- After a new config file has been created (by the skill or manually), **always suggest the `#config-tester` skill** as a next step:
  > 💡 You can verify the new config works with the `#config-tester` skill — it runs a quick smoke test and auto-fixes common issues.

## README.md Maintenance Rules

> **For Copilot:** when editing `README.md`, always keep the following sections in sync with the actual codebase:
>
> 1. **CLI flags table** — must match every `parser.add_argument(...)` call in `cli.py`'s `parse_args()` function. Check flag names, short aliases, metavar strings, and descriptions. Flag names must follow the `--<section>-<field>` naming rule (see below).
> 2. **"Supported backends" table** — must match the `_REGISTRY` dict in `pipelines/__init__.py`.
>
> Never update the README without first reading the current `cli.py` `parse_args()`, and `configs/` directory listing.
>
> **Do NOT maintain a list of existing config files in `README.md` or anywhere else.** The web dashboard populates its dropdown dynamically via `GET /api/configs` — no static list is needed or desired.

## configs/CONFIGS.md Maintenance Rules

`configs/CONFIGS.md` is a **pure reference document** for the config file format. It must contain:

- The full JSON structure example (with all sections)
- Top-level fields table
- **CLI override flag table** — every CLI flag that overrides a config value, with the exact flag name, the config key path it overrides, and its type
- Per-section field tables: `model`, `lora`, `generation`, `system`, `notes`
- `_hint` convention explanation with example
- `_comment*` convention explanation with example
- The "Adding a new config" checklist

**It must NOT contain** a list of existing `.json` files. Never add, update, or remove config file entries in `CONFIGS.md`. When adding a new field to any section or a new pipeline feature, update the corresponding section table in `CONFIGS.md`.

## CLI Flag ↔ Config Key Naming Rule (CRITICAL)

**CLI flags must mirror the config file key path, using hyphens as separators.**

The pattern is: `--<section>-<field>` = `section.field` in the config JSON.

| CLI flag | → | Config key |
|---|---|---|
| `--model-repo` | → | `model.repo` |
| `--model-gguf-file` | → | `model.gguf_file` |
| `--lora-repo` | → | `lora.repo` |
| `--lora-strength` | → | `lora.strength` |
| `--cfg-scale` | → | `generation.cfg_scale` |
| `--steps` | → | `generation.steps` |
| `--output-dir` | → | `system.output_dir` |
| `--cache-dir` | → | `system.cache_dir` |

**Rules:**
1. When adding a new config file field, add a matching CLI flag with the same name pattern (if a CLI override makes sense).
2. When renaming a config key, rename the CLI flag, `EnqueueRequest` field, and HTML form field to match.
3. `EnqueueRequest` field names in `batch/api/jobs.py` must match the CLI flag names exactly (underscores instead of hyphens).
4. HTML form input IDs in `batch/static/index.html` use the `f-` prefix + the CLI flag name (e.g. `f-model-repo`, `f-cfg-scale`).
5. **Always update `configs/CONFIGS.md` CLI override table** when any flag is added, removed, or renamed.

## Config file format (NEW — v2)

Config files use a **nested structure** with four functional sections plus a notes block. All new configs must use this format. The old flat format (with `model_id`, `lora_id`, etc.) is supported via a backward-compat shim during migration only.

```json
{
    "description": "Short label for dashboard",
    "backend": "flux",

    "model": {
        "repo": "...",
        "gguf_file": "...",
        "components_repo": "...",
        "_hint": "Tooltip for the model section in the GUI"
    },

    "lora": {
        "repo": "...",
        "file": "...",
        "strength": 0.9,
        "trigger": "...",
        "_hint": "Tooltip for the LoRA section in the GUI"
    },

    "generation": {
        "steps": 20,
        "cfg_scale": 3.5,
        "width": 1024,
        "height": 1024,
        "max_prompt_tokens": 128,
        "seed": null,
        "_hint": "Tooltip for the generation section in the GUI"
    },

    "system": {
        "cpu_offload": false,
        "cache_dir": "models",
        "output_dir": "outputs",
        "_hint": "Tooltip for the system section in the GUI"
    },

    "notes": {
        "about": "What this model generates — shown as grey info box in the GUI",
        "prompt_guide": "Trigger words, example prompts — shown as collapsible help box",
        "warnings": "Must-know constraints — shown as yellow warning box"
    }
}
```

**Key rules for new configs:**
- `lora` section: omit entirely if no LoRA is used
- `_hint` in each section: describe the most important constraint or value range
- `_comment*` keys: free-form author notes, stripped by parser, never shown in GUI
- `notes.warnings`: always set when `system.cpu_offload` must be `false` or `cfg_scale` must be `0.0`
- GGUF models: `model.gguf_file` + `model.components_repo` required; `system.cpu_offload` must be `false`

## Batch Queue System — Maintenance Rules

When editing the batch section of `README.md`, keep in sync:
1. **Components table** — must match the actual files in `batch/`
2. **REST API table** — must match every `@app.*` route in `batch/server.py`
3. **Worker flags table** — must match `parser.add_argument(...)` calls in `batch/worker.py`'s `main()`
4. **Job statuses table** — must match the status values used in `batch/queue.py`

### Web UI Config Dropdown — Critical Rule
The config dropdown in the web dashboard is **dynamically populated** via `GET /api/configs`.
The server reads `configs/*.json` at request time — there is **no hardcoded list** in `_HTML`.
**When adding a new config file:** just drop the `.json` in `configs/` — the UI picks it up automatically.
**Never** add `<option>` tags to the `_HTML` string in `server.py` for configs.
