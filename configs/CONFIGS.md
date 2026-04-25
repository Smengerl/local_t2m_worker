# Configuration Reference

Config files are JSON files passed via `--config` / `-c`. They describe which model, LoRA, adapter, and generation parameters to use. CLI flags always override config values; config values override built-in defaults.

`_hint` keys and `notes` fields are ignored by the generation pipeline and exist solely for the GUI and documentation.

`_comment*` keys on any level are ignored entirely and exist for the config author only (not rendered in the GUI).

---

## CLI override flags

CLI flags mirror the config file key paths (section.field) using the same names, flattened with a hyphen.  
**Rule: CLI flag name = `--<section>-<field>` whenever a direct config key exists.**

| CLI flag | Config key | Type | Description |
|---|---|---|---|
| `--model-repo REPO_ID` | `model.repo` | string | Base model HF repo ID or local path |
| `--model-gguf-file FILE` | `model.gguf_file` | string | GGUF weight filename inside the model repo |
| `--lora-repo REPO_ID` | `lora.repo` | string | LoRA weights HF repo ID or local path |
| `--lora-strength FLOAT` | `lora.strength` | float | LoRA blend strength `0.0`–`1.0` |
| `--steps N` | `generation.steps` | int | Number of denoising steps |
| `--cfg-scale FLOAT` | `generation.cfg_scale` | float | CFG / guidance scale |
| `--output-dir DIR` | `system.output_dir` | string | Directory for generated images |
| `--cache-dir DIR` | `system.cache_dir` | string | Directory for downloaded model weights |

> **Width / height** can only be overridden via the web API (`width`, `height` fields in `POST /api/jobs`) or by editing the config file directly. There are no `--width` / `--height` CLI flags.

---

## Config file structure

```json
{
    "description": "Human-readable label shown in the dashboard dropdown",
    "backend": "flux",

    "model": {
        "repo": "city96/FLUX.1-dev-gguf",
        "gguf_file": "flux1-dev-Q8_0.gguf",
        "components_repo": "black-forest-labs/FLUX.1-dev",
        "_hint": "Tooltip shown next to the Model section header in the GUI"
    },

    "lora": {
        "repo": "some-org/some-lora",
        "file": "lora-weights.safetensors",
        "strength": 0.9,
        "trigger": "my trigger word",
        "_hint": "Tooltip shown next to the LoRA section header in the GUI"
    },

    "generation": {
        "steps": 20,
        "cfg_scale": 3.5,
        "width": 1024,
        "height": 1024,
        "max_prompt_tokens": 128,
        "seed": null,
        "_hint": "Tooltip shown next to the Generation section header in the GUI"
    },

    "system": {
        "cpu_offload": false,
        "cache_dir": "models",
        "output_dir": "outputs",
        "_hint": "Tooltip shown next to the System section header in the GUI"
    },

    "notes": {
        "about": "General description of the model and the kind of images it produces.",
        "prompt_guide": "Prompting tips: trigger words, style keywords, negative prompt suggestions.",
        "warnings": "Any constraints the user must know: incompatible settings, memory requirements, etc."
    }
}
```

---

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `description` | string | ✅ | Short label — shown in the dashboard dropdown and logs |
| `backend` | string | ✅ | Selects the pipeline backend: `"sd"` / `"sdxl"` / `"sd3"` / `"flux"` / `"zimage"` / `"qwen"` |
| `model` | object | ✅ | Model source — see section below |
| `lora` | object | — | LoRA weights — omit entirely if no LoRA is used |
| `generation` | object | — | Generation parameters — all fields optional, fall back to defaults |
| `system` | object | — | Paths and memory settings — all fields optional |
| `notes` | object | — | Free-text fields rendered as info/guide/warning boxes in the GUI |

---

## `model`

| Field | Type | Default | Description |
|---|---|---|---|
| `repo` | string | **(required)** | HuggingFace repo ID or local path. For GGUF: the repo containing the `.gguf` file (e.g. `city96/FLUX.1-dev-gguf`). For standard models: the full model repo (e.g. `black-forest-labs/FLUX.1-dev`). |
| `gguf_file` | string | `null` | **GGUF only.** Exact filename of the `.gguf` transformer weights inside `repo` (e.g. `flux1-dev-Q8_0.gguf`). When set, triggers GGUF loading. |
| `components_repo` | string | `null` | **GGUF only.** HF repo that supplies VAE, text encoders, scheduler, and tokenizers (e.g. `black-forest-labs/FLUX.1-dev`). Required when `gguf_file` is set. |
| `file` | string | `null` | Single-file checkpoint filename — loaded via `from_single_file()`. Only used when no `lora` section is present. |
| `_hint` | string | — | Tooltip displayed next to the Model section in the GUI. Not used by the pipeline. |

> **GGUF loading:** the pipeline loads the transformer from `repo` + `gguf_file`, then loads all other components from `components_repo`. Both fields are required together.

---

## `lora` *(omit entirely if unused)*

| Field | Type | Default | Description |
|---|---|---|---|
| `repo` | string | **(required)** | HuggingFace repo ID or local path for the LoRA weights |
| `file` | string | `null` | Specific `.safetensors` filename inside `repo` — required for multi-file repos (e.g. `ByteDance/Hyper-SD`) |
| `strength` | float | `0.9` | LoRA blend strength `0.0`–`1.0` |
| `trigger` | string | `null` | Trigger word required by the LoRA. Automatically prepended to the prompt with a warning if missing. |
| `_hint` | string | — | Tooltip displayed next to the LoRA section in the GUI. Not used by the pipeline. |

---

## `generation` *(all fields optional)*

| Field | Type | Default | Description |
|---|---|---|---|
| `steps` | int | `30` | Number of denoising steps — more = better quality, slower |
| `cfg_scale` | float | `7.5` | CFG scale — prompt adherence. Set to `0.0` for CFG-distilled models (FLUX.1-schnell, SDXL-Turbo, Z-Image-Turbo). |
| `cfg_scale_secondary` | float | `null` | Secondary CFG scale used by some backends (e.g. Qwen-Image) |
| `width` | int | `1024` | Output image width in pixels |
| `height` | int | `1024` | Output image height in pixels |
| `max_prompt_tokens` | int | `null` | Maximum T5 encoder token length (FLUX backends). `128` saves ~1 GB RAM vs `256`. Use `256` only for very long prompts. |
| `seed` | int | `null` | Fixed RNG seed for reproducible results. `null` = random each run. |
| `_hint` | string | — | Tooltip displayed next to the Generation section in the GUI. Not used by the pipeline. |

---

## `system` *(all fields optional)*

| Field | Type | Default | Description |
|---|---|---|---|
| `cpu_offload` | bool | `false` | Offload pipeline sub-modules to CPU between operations — reduces peak VRAM by ~50 %. Recommended for SDXL / FLUX / SD3 on ≤ 16 GB RAM. **Must be `false` for GGUF configs** (incompatible with quantised tensors). |
| `cache_dir` | string | `"models"` | Directory for downloaded model weights |
| `output_dir` | string | `"outputs"` | Directory for generated images |
| `_hint` | string | — | Tooltip displayed next to the System section in the GUI. Not used by the pipeline. |

---

## `notes` *(all fields optional)*

All three fields are plain strings. The GUI renders each one in a distinct visual style when present.

| Field | GUI rendering | Content |
|---|---|---|
| `about` | Grey info box at the top | General description of the model, style, and generated images |
| `prompt_guide` | Collapsible prompt-help box | Trigger words, example prompts, style keywords, negative prompt suggestions |
| `warnings` | Yellow/orange warning box | Must-know constraints: incompatible settings, memory requirements, model limitations |

---

## `_hint` convention

Each section (`model`, `lora`, `generation`, `system`) may contain a `_hint` string. The GUI shows it as a tooltip (ℹ️ icon) next to the section header. Describe the most important constraint or value range for that section.

```json
"generation": {
    "steps": 4,
    "cfg_scale": 0.0,
    "_hint": "FLUX.1-schnell is CFG-distilled — cfg_scale must be 0.0. Steps 1–8, sweet spot at 4."
}
```

---

## `_comment*` convention

Keys starting with `_comment` are stripped by the parser and never reach the pipeline or GUI. Use them freely for author notes in the JSON file itself.

```json
"model": {
    "repo": "city96/FLUX.1-dev-gguf",
    "gguf_file": "flux1-dev-Q8_0.gguf",
    "_comment": "Q8_0 keeps near-bfloat16 quality with trivial dequantisation overhead on MPS."
}
```

---

## Adding a new config

1. Copy an existing config with the same `backend`.
2. Set `model.repo` and optionally the `lora` section.
3. For GGUF models: set `model.gguf_file` + `model.components_repo`; set `system.cpu_offload` to `false`.
4. Adjust `generation.steps`, `generation.cfg_scale`, `generation.width`, `generation.height` per the model card.
5. For non-GGUF SDXL / FLUX / SD3 on Mac: set `system.cpu_offload` to `true`.
6. Add `notes.about`, `notes.prompt_guide`, and `notes.warnings` for documentation and GUI hints.
7. Save to `configs/your_name.json` — picked up automatically by the web dashboard and `run.sh`.

Configs in sub-folders (`nfsw/`, `high_memory/`) are excluded from the web dashboard dropdown.
