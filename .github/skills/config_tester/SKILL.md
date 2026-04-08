---
name: config-tester
description: >
  Tests a given configs/*.json file by running generate.py with a
  model-appropriate example prompt. Monitors for startup errors and kills
  the process as soon as the first denoising step fires (proving the pipeline
  is working). Diagnoses and fixes any errors found in the pipeline code or
  config before reporting the result.
allowed-tools: python
---

## Goal

Given a config file path (e.g. `configs/sdxl_graffiti_lora.json`), verify
that it loads correctly and that inference actually starts — without waiting
for a full image to be generated.

**Success criterion:** The process prints `Step 1/` (or the equivalent first
progress line) and is then terminated. This proves the model loaded, the
pipeline initialised, and denoising began.

---

## Step 1 — Read and parse the config

1. Open the JSON file and parse it (strip `_comment*` keys).
2. Note the values:
   - `pipeline_type`
   - `model_id`
   - `lora_id` (may be `null`)
   - `trigger_word` (may be `null`)
   - `num_inference_steps`
   - `guidance_scale`
   - `width` × `height`
   - `sequential_cpu_offload`
3. Verify the JSON is valid and all mandatory keys are present
   (`pipeline_type`, `model_id`). If not, report the issue immediately and
   stop.

---

## Step 2 — Compose a model-appropriate example prompt

Choose a short prompt that matches the style the model was trained for.
Use the `description`, `_comment`, `_comment_prompt`, and `trigger_word`
fields from the config as guidance.

### Prompt selection rules

| Condition | Prompt strategy |
|---|---|
| `trigger_word` is set | **Prepend the trigger word**: `"<trigger_word> <generic subject>"` |
| `pipeline_type` is `sd` / `sdxl` / `sd3` — generic/realistic | `"a photorealistic portrait of a woman in a park, golden hour lighting"` |
| Style suggests illustration / comic / anime | `"a warrior in armor, dramatic lighting, detailed illustration"` |
| Style suggests pixel art | `"a small pixel art character on a grassy hill"` |
| Style suggests line art / coloring book | `"a cat sitting on a window sill, clean line art, no shading"` |
| Style suggests graffiti / street art | `"a graffiti mural of a wolf on a brick wall"` |
| Style suggests manga / black & white | `"a samurai in a bamboo forest, black and white manga style"` |
| `pipeline_type` is `zimage` | `"a close-up portrait of a person, sharp focus, photorealistic"` |
| `pipeline_type` is `qwen` | `"a scenic landscape at sunrise, soft light, high detail"` |
| `pipeline_type` is `anima` | `"a fantasy character in an enchanted forest, anime style"` |

Keep the prompt under 30 words to minimise generation time.

---

## Step 3 — Run generate.py and monitor output

### 3a — Build the command

```bash
python generate.py -c <config_path> "<prompt>"
```

Use the exact config path as given. Do **not** add `--steps`, `--guidance-scale`,
or any other override flags — the point is to test the config as-is.

### 3b — Run with a timeout

Run the command in a terminal. The process will:
1. Print the config summary block.
2. Log model loading lines (`Loading SD base model: ...`, `Loading LoRA weights: ...`, etc.)
3. Begin denoising and emit progress lines.

**Abort condition:** As soon as you see any of these patterns in stdout/stderr,
kill the process (Ctrl+C / SIGINT):

| Pattern | Meaning |
|---|---|
| `Step 1/` | First denoising step fired → **SUCCESS** |
| `100%\|` followed by `1/` in the tqdm bar | First step in tqdm format → **SUCCESS** |
| Progress callback output matching `step=1` | First callback → **SUCCESS** |
| Any line starting with `Traceback` | Python exception → **ERROR** |
| `Error` / `error:` / `KeyError` / `ValueError` / `RuntimeError` | Startup failure → **ERROR** |
| `CUDA out of memory` / `MPS out of memory` | OOM → **ERROR** |
| Process exits with non-zero code before first step | Startup failure → **ERROR** |

**Maximum wait time:** 10 minutes. If no progress and no error appear within
that window, treat it as a hang and kill the process.

### 3c — Capture and parse the output

Collect the full stdout+stderr output. You will need it for diagnosis in
case of errors.

---

## Step 4 — Interpret the result

### Case A — SUCCESS (first step fired)

Report:

```
✅ Config test PASSED: configs/<name>.json
   Pipeline started successfully — first denoising step reached.
   Model loaded: <model_id>
   LoRA loaded: <lora_id or 'none'>
   Process terminated after step 1 (as expected).
```

No further action needed.

---

### Case B — ERROR (exception or non-zero exit)

Analyse the error output carefully. Follow the diagnosis tree below.

#### Error diagnosis tree

**`KeyError: 'pipeline_type'`**
→ Config is missing `pipeline_type`. Add it.

**`ValueError: Unknown pipeline_type '<type>'`**
→ The `pipeline_type` value in the config is not registered in `_REGISTRY`
  inside `pipelines/__init__.py`.
→ Either fix the typo in the config, or create a new pipeline (follow the
  config-builder skill's Step 3c).

**`FileNotFoundError` / `OSError: No such file or directory` for a model path**
→ `model_id` or `lora_id` is wrong. Check the HF repo ID (case-sensitive).
  Search `huggingface.co` to verify it exists.

**`huggingface_hub.utils._errors.GatedRepoError` / `401 Unauthorized`**
→ The model is gated. The user needs to accept the licence at
  `https://huggingface.co/<model_id>` and be logged in (`huggingface-cli login`).
→ Add `"_comment_gated"` to the config if it is missing.

**`RuntimeError: size mismatch` during LoRA loading**
→ The LoRA was trained on a different base than `model_id`.
  Cross-check the LoRA model card for the correct base model.

**`ImportError` / `ModuleNotFoundError` for a diffusers class**
→ The pipeline uses a diffusers class not yet available in the installed
  version. Check `requirements.txt` and upgrade if needed:
  `pip install --upgrade diffusers`.

**`torch.cuda.OutOfMemoryError` / `MPS backend out of memory`**
→ The model exceeds available VRAM.
→ Set `"sequential_cpu_offload": true` in the config and retry.

**`RuntimeError: Expected all tensors to be on the same device`**
→ Device placement error, often triggered when `sequential_cpu_offload` is
  `false` on a machine with limited memory. Set it to `true`.

**`ValueError: ... guidance_scale ... must be`**
→ CFG-distilled model (e.g. Z-Image-Turbo, FLUX-schnell) requires
  `guidance_scale: 0.0`. Fix the config.

**`weight_name` / `from_single_file` related `404` / `EntryNotFoundError`**
→ The `weight_name` field in the config is wrong. Open the HF repo's
  "Files & versions" tab, find the exact filename, and update `weight_name`.

**Any other `RuntimeError` or `AttributeError` in a pipeline file**
→ Read the full traceback. Identify which file and line is failing.
  Open that file, understand the error, and apply a minimal fix.
  Common causes:
  - Pipeline passing an unsupported kwarg to a diffusers class
  - Diffusers API change (check diffusers changelog for the installed version)
  - LoRA loading API difference between diffusers versions

---

## Step 5 — Apply fixes and retry

After identifying the root cause:

1. **Config-only fix** (wrong value, missing key, typo):
   - Edit the config JSON directly.
   - Re-run the test from Step 3.

2. **Pipeline code fix** (bug in `pipelines/*.py`):
   - Edit the pipeline file with a minimal, targeted fix.
   - Explain the change in your response.
   - Re-run the test from Step 3.

3. **New pipeline required** (unknown `pipeline_type`):
   - Follow the config-builder skill's Step 3c to create and register it.
   - Re-run the test from Step 3.

4. **Environment issue** (missing package, outdated diffusers):
   - Run `pip install --upgrade <package>` or `pip install -r requirements.txt`.
   - Re-run the test from Step 3.

**Retry limit:** Attempt fixes up to **3 times**. If still failing after 3
attempts, stop and report all findings to the user with a clear summary of
what was tried and what the remaining blocker is.

---

## Step 6 — Report the final result

Always produce a structured summary:

```
## Config test result: configs/<name>.json

**Status:** ✅ PASSED  /  ❌ FAILED

**Prompt used:** "<prompt>"

**Pipeline:** <pipeline_type> → <class name>
**Model:** <model_id>
**LoRA:** <lora_id or 'none'>

**Outcome:**
<One paragraph: what happened, what was fixed (if anything), final state.>

**Fixes applied:** (list any changes made to config or pipeline files)
  - <file>: <description of change>

**To run a full generation:**
  ./run.sh -c <config_path> "<prompt>"
```
