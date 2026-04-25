# Text-to-Image Local Inference Project

Python project for running Hugging Face text-to-image models (Stable Diffusion) locally using the `diffusers` library.

> For usage, CLI flags, config keys, available presets, and batch system documentation see **README.md**.

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
