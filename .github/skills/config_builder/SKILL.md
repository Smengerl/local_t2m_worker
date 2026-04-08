---
name: hugging-face-config-builder
description: >
  Creates a JSON config file for a Hugging Face text-to-image model and
  documents it in README.md. Researches model parameters from the HF Hub,
  chooses the correct pipeline_type, adds rich _comment fields, and checks
  whether a new pipeline class is required.
allowed-tools: web-search
---

## Goal

Given a Hugging Face model name or URL, produce a complete, runnable
`configs/<slug>.json` file for this project's text-to-image generation system
and update `README.md` accordingly.

---

## Step 1 — Resolve the model identifier

1. If the user supplies a URL like `https://huggingface.co/author/model-name`,
   extract the HF repo ID `author/model-name`.
2. If only a short name is given, assume it may be a repo ID or search term.
   Use a web search (`site:huggingface.co <name>`) to find the canonical repo.
3. Note down:
   - **repo ID** (e.g. `stabilityai/stable-diffusion-xl-base-1.0`)
   - Whether this is a **base model**, a **LoRA adapter**, a **DreamBooth checkpoint**, or another adapter type.

---

## Step 2 — Research the model

Fetch the model card from `https://huggingface.co/<repo-id>` (and the
"Files & versions" tab if helpful). Extract the following information. If a
value is not stated, use the heuristics in parentheses:

| Field to find | Where to look / heuristic |
|---|---|
| **Architecture** | Model card tags, name, or "Model Description" — e.g. SD 1.x, SD 2.x, SDXL, SD3, FLUX, DiT, Z-Image, Qwen-Image, Cosmos, AnimaYume |
| **Base model** | "Base model:" line or tags. Needed when model is a LoRA or DreamBooth fine-tune. |
| **Trigger word(s)** | "Trigger word", "Activation word", or "Usage" section. `null` if none. |
| **Recommended steps** | Model card example code or "Usage" section. Fallback: architecture defaults below. |
| **Recommended CFG / guidance_scale** | Same sources. Fallback: architecture defaults below. |
| **Native resolution** | Tags (`512x512`, `1024x1024`) or training details. |
| **LoRA scale** | Model card, otherwise `0.9`. |
| **Gated / restricted** | Model card header — note this in a `_comment`. |
| **Special kwargs** | e.g. `true_cfg_scale` for CFG-distilled models, `weight_name` for single-file checkpoints. |

### Architecture defaults (use when model card is silent)

| Architecture | `pipeline_type` | Steps | CFG | Native size |
|---|---|---|---|---|
| SD 1.x (1.4 / 1.5) | `sd` | 30 | 7.5 | 512×512 |
| SD 2.x (2.0 / 2.1 base) | `sd` | 30 | 7.5 | 512×512 |
| SD 2.x (2.1 full) | `sd` | 30 | 7.5 | 768×768 |
| SDXL 1.0 | `sdxl` | 30 | 7.5 | 1024×1024 |
| SD 3 / SD 3.5 | `sd3` | 28 | 7.0 | 1024×1024 |
| FLUX (dev / schnell) | `flux` | 20–28 | 3.5 | 1024×1024 |
| Z-Image-Turbo | `zimage` | 12 | 0.0 | 1024×1024 |
| Qwen-Image | `qwen` | 20 | — (`true_cfg_scale` 4.0) | 1328×1328 |
| AnimaYume / Cosmos | `anima` | 30 | 7.5 | 1024×1024 |

---

## Step 3 — Determine the pipeline_type and check whether a new pipeline is needed

### 3a — Match to an existing pipeline

Look at `pipelines/__init__.py`. The current `_REGISTRY` maps:

```python
_REGISTRY = {
    "sd":     "pipelines.sd_pipeline.StableDiffusionBackend",
    "sdxl":   "pipelines.sd_pipeline.StableDiffusionBackend",
    "sd3":    "pipelines.sd_pipeline.StableDiffusionBackend",
    "anima":  "pipelines.anima_pipeline.AnimaPipeline",
    "zimage": "pipelines.zimage_pipeline.ZImageBackend",
    "qwen":   "pipelines.qwen_pipeline.QwenImageBackend",
}
```

`StableDiffusionBackend` (in `sd_pipeline.py`) handles `sd`, `sdxl`, and `sd3`
by selecting the matching diffusers class internally.

### 3b — Decide if the architecture fits an existing pipeline

| Model architecture | Use existing pipeline? | Notes |
|---|---|---|
| SD 1.x / 2.x | ✅ `sd` | Direct `from_pretrained` or `from_single_file` |
| SDXL 1.0 | ✅ `sdxl` | `StableDiffusionXLPipeline` auto-selected |
| SD 3 / 3.5 | ✅ `sd3` | `StableDiffusion3Pipeline` auto-selected |
| FLUX (dev/schnell) | ❌ **New pipeline needed** | `FluxPipeline` from diffusers |
| LCM / Lightning | ❌ **New pipeline** | Scheduler swap required |
| Kandinsky | ❌ **New pipeline** | Two-stage architecture |
| Other novel arch | ❌ **New pipeline** | Evaluate case-by-case |

### 3c — If a new pipeline IS required

1. Create `pipelines/<arch>_pipeline.py` that:
   - Inherits from `BasePipeline` (see `pipelines/base.py`)
   - Implements `generate(prompt, negative_prompt, progress_callback) -> Image`
   - Follows the device selection pattern (`self._get_device()`)
   - Applies `sequential_cpu_offload` if `self.sequential_cpu_offload` is True
   - Logs key steps with `self._log(...)`
2. Register it in `_REGISTRY` in `pipelines/__init__.py`.
3. Add it to the **"Supported backends" table** in `README.md`.
4. Note this action explicitly in your response to the user.

---

## Step 4 — Determine the config file name

Use the pattern `<arch>_<short-description>.json`:

- Architecture prefix: `sd15_`, `sd21_`, `sdxl_`, `sd3_`, `flux_`, `zimage_`, `qwen_`, `anima_`
- Short description in `snake_case`: model name, style, or LoRA name (max ~25 chars)
- Examples: `sdxl_realism_engine_lora.json`, `flux_schnell_default.json`, `sd15_van_gogh_lora.json`

Save the file to `configs/<slug>.json`.

---

## Step 5 — Build the JSON config

Produce a JSON object with **all** keys below. All `_comment*` keys are
stripped at runtime (keys starting with `_` are ignored by `cli.py`), so use
them freely to document valid ranges, gotchas, and usage hints for the user.

### Required keys (in this order)

```jsonc
{
    "description": "<One-line human label shown at startup>",
    "_comment": "<Architecture / training summary — what makes this model unique>",

    // ── Trigger word ────────────────────────────────────────────────────────
    // Include _comment_trigger only if trigger_word is not null.
    "_comment_trigger": "TRIGGER WORD: include '<word>' in every prompt.",
    "_comment_prompt":  "Example prompts: ...",

    // ── Negative prompt hint ─────────────────────────────────────────────────
    "_comment_negative": "Recommended negative: '...'",

    // ── Steps ────────────────────────────────────────────────────────────────
    "_comment_steps": "Recommended range X–Y. Default Z per model card.",

    // ── CFG ──────────────────────────────────────────────────────────────────
    "_comment_cfg": "CFG X–Y recommended. Notes on what higher/lower values do.",

    // ── LoRA scale (include only when lora_id is not null) ───────────────────
    "_comment_lora_scale": "Range 0.0–1.0. 0.9 safe default. Effect at extremes: ...",

    // ── Resolution ───────────────────────────────────────────────────────────
    "_comment_size": "Native resolution NxM. Notes on other supported sizes.",

    // ── Memory ───────────────────────────────────────────────────────────────
    "_comment_offload": "Peak VRAM ~X GB. Enable sequential_cpu_offload on ≤Y GB unified memory.",

    // ── Gated model (include only if model requires HF login) ────────────────
    "_comment_gated": "Gated model — requires HF account agreement at huggingface.co/<repo-id>.",

    // ── Functional keys ──────────────────────────────────────────────────────
    "pipeline_type": "<type>",
    "model_id": "<hf-repo-id>",
    "adapter_id": null,
    "lora_id": "<hf-repo-id or null>",
    "lora_scale": 0.9,
    "trigger_word": "<word or null>",
    "num_inference_steps": <int>,
    "guidance_scale": <float>,
    "width": <int>,
    "height": <int>,
    "sequential_cpu_offload": <bool>
}
```

### Optional keys — add only when needed

| Key | When to add |
|---|---|
| `"weight_name": "<filename.safetensors>"` | Model is a single-file checkpoint (use `from_single_file`). Look for `.safetensors` / `.ckpt` files in the repo's "Files" tab. |
| `"true_cfg_scale": <float>` | CFG-distilled models (Qwen-Image, Z-Image-Turbo) that use a separate true-CFG parameter instead of `guidance_scale`. |
| `"seed": <int>` | Only when the model card specifically recommends a seed for reproducibility. |

### Rules for specific architectures

- **SD 1.x / 2.x DreamBooth** (base model, no LoRA): set `lora_id: null`, put the fine-tune repo as `model_id`. If the repo contains a single `.safetensors` file, add `weight_name`.
- **LoRA on top of a base model**: set `model_id` to the base, `lora_id` to the LoRA repo.
- **SDXL + LoRA**: `sequential_cpu_offload: true` by default (SDXL is ~10 GB; 16 GB unified memory machines will OOM without it).
- **SD3**: `sequential_cpu_offload: true` by default (~10 GB peak VRAM). Use `pipeline_type: "sd3"`.
- **CFG-distilled / turbo** (Z-Image, FLUX-schnell): `guidance_scale: 0.0` or very low (1.0). Note this prominently in `_comment_cfg`.

---

## Step 6 — Update README.md

After saving the config file, open `README.md` and update **two sections**:

### 6a — "Configuration files" table

Add a new row. Match the existing table format:

```markdown
| `configs/<slug>.json` | `<pipeline_type>` | `<description value>` |
```

Keep the rows sorted: `sd15_*` → `sd21_*` → `sdxl_*` → `sd3_*` → `flux_*` → `zimage_*` → `qwen_*` → `anima_*`.

### 6b — "Supported backends" table (only if a new pipeline was added)

Add a row for the new `pipeline_type`:

```markdown
| `<type>` | `<NewClassName>` | diffusers |
```

---

## Step 7 — Report to the user

Summarise what was done:

1. Config file created: `configs/<slug>.json`
2. Key parameter choices and sources (model card link, pages consulted)
3. Whether a new pipeline was required and what was created/registered
4. Any caveats (gated model, unusual CFG, memory requirements)
5. README.md sections updated
6. A quick-start example command:
   ```
   ./run.sh -c ./configs/<slug>.json "YOUR PROMPT HERE"
   ```

---

## Worked example — adding FLUX.1-schnell

**User input:** `black-forest-labs/FLUX.1-schnell`

**Research findings:**
- Architecture: FLUX (novel DiT, not SD-family)
- Not in `_REGISTRY` → new pipeline required
- Steps: 4 (distilled, very fast)
- CFG: 0.0 (guidance-distilled, no classifier-free guidance)
- Native resolution: 1024×1024
- No trigger word, not gated

**Actions:**
1. Create `pipelines/flux_pipeline.py` with `FluxBackend(BasePipeline)` using `diffusers.FluxPipeline`.
2. Register `"flux": "pipelines.flux_pipeline.FluxBackend"` in `pipelines/__init__.py`.
3. Write `configs/flux_schnell_default.json`:
   ```json
   {
       "description": "FLUX.1-schnell (Black Forest Labs) — ultra-fast 4-step distilled text-to-image",
       "_comment": "Guidance-distilled FLUX model. 4 steps, no negative prompt, guidance_scale must be 0.0.",
       "_comment_steps": "4 steps is the official recommendation. Range 1–8; more steps do not improve quality.",
       "_comment_cfg": "MUST be 0.0 — this model is guidance-distilled (no CFG). Any other value degrades output.",
       "_comment_size": "Native 1024x1024. Any multiple of 64 up to ~2 MP is supported.",
       "_comment_offload": "~23 GB peak VRAM. sequential_cpu_offload strongly recommended on ≤24 GB.",
       "pipeline_type": "flux",
       "model_id": "black-forest-labs/FLUX.1-schnell",
       "adapter_id": null,
       "lora_id": null,
       "lora_scale": 0.9,
       "trigger_word": null,
       "num_inference_steps": 4,
       "guidance_scale": 0.0,
       "width": 1024,
       "height": 1024,
       "sequential_cpu_offload": true
   }
   ```
4. Update README.md config table and backends table.
