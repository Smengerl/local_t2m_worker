"""
CLI argument parsing and configuration loading for the text-to-image generator.

Responsibilities:
  - Define and parse all command-line arguments (parse_args)
  - Load JSON config files into typed PipelineConfig objects (PipelineConfig.from_json)
  - Apply CLI flag overrides on top of the loaded config (build_config)
  - Print a human-readable summary of the effective configuration (print_config)

Required CLI parameters:
  --config / -c   Config file that specifies the backend and model.
  PROMPT          Positional text prompt.

Output (mutually exclusive, both optional):
  --output / -o   Explicit output PNG file path.
  --output-dir    Directory; filename is assigned automatically (YYYYMMDD_HHMMSS.png).
  (neither)       Timestamped file placed in the current working directory.

All other parameters are optional overrides.  Default values for generation
parameters (steps, cfg_scale, width, height) are NOT defined here — they live
in the respective backend pipeline class (``GENERATION_DEFAULTS``).

Nothing in this module imports torch or diffusers — it is pure Python stdlib.
"""

import argparse
import os
from datetime import datetime
from typing import Optional

from pipeline_config import PipelineConfig


def parse_args() -> argparse.Namespace:
    """Define and parse all CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using Stable Diffusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate.py -c configs/flux_schnell.json -o out.png 'a sunset'\n"
            "  python generate.py -c configs/sdxl_graffiti_lora.json"
            " -o outputs/dragon.png 'graffiti dragon'\n"
            "  python generate.py -c configs/sd15_default.json"
            " -o cat.png --steps 50 --cfg-scale 8.0 'a cat'\n"
        ),
    )

    # ── Positional: prompt (always last, no flag needed) ──────────────────────
    parser.add_argument(
        "prompt",
        metavar="PROMPT",
        help="Text prompt describing the image to generate.",
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", "-c",
        required=True,
        metavar="FILE",
        help=(
            "Path to a JSON config file specifying the backend and model "
            "(see configs/ for examples). CLI flags take precedence over "
            "values in the config file."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help=(
            "Output file path (PNG). "
            "Mutually exclusive with --output-dir. "
            "If neither is given, a timestamped file is placed in the current directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help=(
            "Directory for the generated image. "
            "A timestamped filename is assigned automatically. "
            "Mutually exclusive with --output."
        ),
    )

    # ── Prompts ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--negative-prompt", "-n",
        default="",
        metavar="TEXT",
        help="Negative prompt (what to avoid in the image). Default: empty.",
    )

    # ── System overrides ──────────────────────────────────────────────────────
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        help=(
            "Directory for downloaded model weights. If not provided, the Hugging Face/diffusers"
            " default cache location is used (e.g. ~/.cache/huggingface), allowing model sharing"
            " between applications."
        ),
    )

    # ── Model / LoRA overrides ────────────────────────────────────────────────
    parser.add_argument(
        "--model-repo",
        metavar="REPO_ID",
        help="Hugging Face base model repo ID or local path. Overrides config model.repo.",
    )
    parser.add_argument(
        "--model-gguf-file",
        metavar="FILENAME",
        help="GGUF weight filename inside the model repo. Overrides config model.gguf_file.",
    )
    parser.add_argument(
        "--lora-repo",
        metavar="REPO_ID",
        help="Hugging Face LoRA weights repo ID or local path. Overrides config lora.repo.",
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        metavar="FLOAT",
        help="LoRA blend strength (0.0–1.0). Overrides config lora.strength.",
    )

    # ── Inference overrides ───────────────────────────────────────────────────
    parser.add_argument(
        "--steps",
        type=int,
        metavar="N",
        help=(
            "Number of inference steps. Overrides config generation.steps. "
            "If unset, the backend's default is used."
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        metavar="FLOAT",
        help=(
            "CFG / guidance scale. Overrides config generation.cfg_scale. "
            "If unset, the backend's default is used."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        metavar="N",
        help="Image width in pixels. Overrides config generation.width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        metavar="N",
        help="Image height in pixels. Overrides config generation.height.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> tuple[PipelineConfig, str, str, str]:
    """Load config file → apply CLI overrides → return a resolved PipelineConfig.

    Args:
        args: Parsed argument namespace returned by ``parse_args()``.

    Returns:
        A tuple of ``(cfg, output_path, effective_prompt, negative_prompt)`` where:

        - *cfg* is a ``PipelineConfig`` with all static pipeline parameters.
          Generation fields (steps, cfg_scale, width, height) may still be
          ``None`` here — the backend resolves them against its own defaults.
        - *output_path* is the resolved path for the output PNG file.
        - *effective_prompt* is the prompt passed to the pipeline (trigger word
          prepended automatically if required).
        - *negative_prompt* is the negative prompt as supplied by the caller.
    """
    # 1. Load config file into typed objects.
    pipeline_cfg = PipelineConfig.from_json(args.config)

    # 2. Apply explicit CLI overrides in one call.
    pipeline_cfg.apply_overrides(
        model_repo=args.model_repo,
        model_gguf_file=args.model_gguf_file,
        lora_repo=args.lora_repo,
        lora_strength=args.lora_strength,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        width=args.width,
        height=args.height,
        cache_dir=args.cache_dir,
    )

    # 3. Trigger-word check — prepend automatically if missing from the prompt.
    effective_prompt: str = args.prompt
    if pipeline_cfg.trigger_word and pipeline_cfg.trigger_word.lower() not in effective_prompt.lower():
        print(
            f"⚠️  Trigger word {pipeline_cfg.trigger_word!r} not found in prompt — "
            f"prepending it automatically."
        )
        effective_prompt = f"{pipeline_cfg.trigger_word} {effective_prompt}"

    negative_prompt: str = args.negative_prompt

    # 4. Resolve output path:
    #    a) explicit --output  → use as-is
    #    b) --output-dir given → place timestamped file inside that directory
    #    c) neither given      → place timestamped file in current directory
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = getattr(args, "output_dir", None) or "."
        output_path = os.path.join(output_dir, f"{timestamp}.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    return pipeline_cfg, output_path, effective_prompt, negative_prompt


def print_config(
    cfg: PipelineConfig,
    output_path: str,
    effective_prompt: str,
    negative_prompt: str,
    original_prompt: str,
) -> None:
    """Print a human-readable summary of the effective configuration.

    Args:
        cfg: PipelineConfig (from build_config).
        output_path: Resolved output file path.
        effective_prompt: Prompt passed to the pipeline (may have trigger word prepended).
        negative_prompt: Negative prompt as supplied by the caller.
        original_prompt: Prompt as supplied by the caller, before trigger-word prepend.
    """
    print("── Prompt ───────────────────────────────────────────────────────")
    print(f"  prompt            : {effective_prompt!r}")
    if effective_prompt != original_prompt:
        print(f"  prompt (original) : {original_prompt!r}")
    print(f"  negative_prompt   : {negative_prompt!r}")
    print("── Output file ──────────────────────────────────────────────────")
    print(f"  output            : {output_path}")
    print("── Pipeline configuration ───────────────────────────────────────")
    if cfg.description:
        print(f"  description       : {cfg.description}")
    print(f"  pipeline_type     : {cfg.pipeline_type}")
    print(f"  model_id          : {cfg.model_id}")
    print(f"  lora_id           : {cfg.lora_id or '(none)'}")
    print(f"  lora_scale        : {cfg.lora_scale}")
    if cfg.trigger_word:
        print(f"  trigger_word      : {cfg.trigger_word!r}")
    steps_str = str(cfg.num_inference_steps) if cfg.num_inference_steps is not None else "(backend default)"
    print(f"  steps             : {steps_str}")
    if cfg.true_cfg_scale is not None:
        print(f"  true_cfg_scale    : {cfg.true_cfg_scale}")
    else:
        cfg_str = str(cfg.guidance_scale) if cfg.guidance_scale is not None else "(backend default)"
        print(f"  guidance_scale    : {cfg_str}")
    w = cfg.width  if cfg.width  is not None else "(backend default)"
    h = cfg.height if cfg.height is not None else "(backend default)"
    print(f"  width x height    : {w} x {h}")
    print(f"  cpu_offload       : {cfg.sequential_cpu_offload}")
    print("─────────────────────────────────────────────────────────────────")
