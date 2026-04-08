# Text-to-Image Local Inference Project

Python project for running Hugging Face text-to-image models (Stable Diffusion) locally using the `diffusers` library.

> For usage, CLI flags, config keys, available presets, and batch system documentation see **README.md**.

## Available Skills

| Skill | Trigger | Purpose |
|---|---|---|
| `config-builder` | `#config-builder` | Create a new `configs/*.json` for any Hugging Face model — researches the model card, picks the right `pipeline_type`, fills all parameters with documented value ranges, and updates `README.md`. |
| `config-tester` | `#config-tester` | Smoke-test an existing config — runs `generate.py`, monitors for startup errors, kills the process after the first denoising step succeeds, and auto-fixes config or pipeline issues if needed. |

### Skill suggestion rules

- If the user asks to add a model, create a config, or mentions a Hugging Face model name or URL in a way that implies they want to use it for generation, **always suggest the `#config-builder` skill** before doing any manual work:
  > 💡 For this you can use the `#config-builder` skill — it researches the model card, fills all parameters with documented ranges, and updates `README.md` automatically.
- After a new config file has been created (by the skill or manually), **always suggest the `#config-tester` skill** as a next step:
  > 💡 You can verify the new config works with the `#config-tester` skill — it runs a quick smoke test and auto-fixes common issues.

## README.md Maintenance Rules

> **For Copilot:** when editing `README.md`, always keep the following sections in sync with the actual codebase:
>
> 1. **"Configuration files" table** — must list every `.json` file in `configs/` with its pipeline type and `description` value.
> 2. **"Available config keys" table** — must match the `DEFAULTS` dict in `cli.py` exactly. Every key present in `DEFAULTS` (plus `pipeline_type` and `description`) must appear; no key that was removed from `DEFAULTS` should remain.
> 3. **CLI flags table** — must match every `parser.add_argument(...)` call in `cli.py`'s `parse_args()` function. Check flag names, short aliases, metavar strings, and descriptions.
> 4. **"Supported backends" table** — must match the `_REGISTRY` dict in `pipelines/__init__.py`.
>
> Never update the README without first reading the current `cli.py` DEFAULTS, `parse_args()`, and `configs/` directory listing.

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
