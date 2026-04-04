"""
CLI argument parsing and configuration loading for the text-to-image generator.

Responsibilities:
  - Define and parse all command-line arguments (parse_args)
  - Load and merge JSON config files with built-in defaults (load_config)
  - Apply CLI flag overrides on top of the loaded config (build_config)
  - Print a human-readable summary of the effective configuration (print_config)

Nothing in this module imports torch or diffusers — it is pure Python stdlib.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Default values ────────────────────────────────────────────────────────────
# Lowest priority: overridden by config file, then by CLI flags.
DEFAULTS: dict = {
    "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    # pipeline_type is intentionally NOT listed here — it must be set
    # explicitly in every config file. Omitting it causes a clear error.
    # Valid values: "sd" | "sdxl" | "anima"
    "adapter_id": None,    # optional: HF repo ID / local path for an adapter (ControlNet, refiner, …)
    "lora_id": None,       # optional: HF repo ID / local path for LoRA weights
    "lora_scale": 0.9,
    "trigger_word": None,  # optional: token that must appear in the prompt for the LoRA/model to activate
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "output_dir": "outputs",
    "cache_dir": "models",
    # Offload model submodules to CPU between steps to save GPU/MPS memory.
    # Slower, but necessary for SDXL on machines with ≤16 GB unified memory.
    # Ignored for the anima pipeline_type.
    "sequential_cpu_offload": False,
}
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: Optional[str]) -> dict:
    """Load a JSON config file and merge with built-in defaults.

    Keys in the file override DEFAULTS; CLI flags (applied later) override both.

    Args:
        config_path: Path to the JSON config file, or None to use only defaults.

    Returns:
        Merged configuration dict.

    Raises:
        FileNotFoundError: If *config_path* is given but does not exist.
    """
    cfg = dict(DEFAULTS)
    if config_path is None:
        return cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open() as fh:
        user_cfg = json.load(fh)

    # Strip annotation-only keys so JSON files may contain "_comment" entries.
    user_cfg = {k: v for k, v in user_cfg.items() if not k.startswith("_")}
    cfg.update(user_cfg)
    return cfg


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
            "  python generate.py 'a sunset over mountains'\n"
            "  python generate.py -c configs/sdxl_graffiti_lora.json"
            " -o outputs/dragon.png 'graffiti dragon'\n"
            "  python generate.py -c configs/sd15_default.json"
            " --steps 50 --guidance-scale 8.0 'a cat'\n"
        ),
    )

    # ── Positional: prompt (always last, no flag needed) ──────────────────────
    parser.add_argument(
        "prompt",
        metavar="PROMPT",
        help="Text prompt describing the image to generate.",
    )

    # ── Config file ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", "-c",
        metavar="FILE",
        help=(
            "Path to a JSON config file (see configs/ for examples). "
            "All keys are optional and override built-in defaults. "
            "CLI flags take precedence over the config file."
        ),
    )

    # ── Prompts ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--negative-prompt", "-n",
        default="",
        metavar="TEXT",
        help="Negative prompt (what to avoid in the image). Default: empty.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help=(
            "Output file path (PNG). "
            "Defaults to <output_dir>/output.png as set in the config."
        ),
    )

    # ── Model / LoRA overrides ────────────────────────────────────────────────
    parser.add_argument(
        "--model-id",
        metavar="REPO_ID",
        help="Hugging Face base model ID or local path. Overrides config.",
    )
    parser.add_argument(
        "--adapter-id",
        metavar="REPO_ID",
        help="Hugging Face adapter (ControlNet, refiner, …) ID or local path. Overrides config.",
    )
    parser.add_argument(
        "--lora-id",
        metavar="REPO_ID",
        help="Hugging Face LoRA weights ID or local path. Overrides config.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        metavar="FLOAT",
        help="LoRA blending scale (0.0–1.0). Overrides config.",
    )

    # ── Inference overrides ───────────────────────────────────────────────────
    parser.add_argument(
        "--steps",
        type=int,
        metavar="N",
        help="Number of inference steps. Overrides config.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        metavar="FLOAT",
        help="Classifier-free guidance scale. Overrides config.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> tuple[dict, str]:
    """Merge defaults → config file → CLI flags into a single config dict.

    Args:
        args: Parsed argument namespace returned by parse_args().

    Returns:
        A tuple of (cfg, output_path) where:
          - cfg is the fully resolved configuration dict.
          - output_path is the resolved path for the output PNG file.
    """
    # 1. Load config file on top of defaults
    cfg = load_config(args.config)

    # 2. Validate mandatory field — every config file must declare pipeline_type
    if "pipeline_type" not in cfg:
        raise KeyError(
            "'pipeline_type' is missing from the config file.\n"
            "Add it to your JSON config, e.g.:\n"
            '  "pipeline_type": "sd"   # or "sdxl" / "anima"'
        )

    # 3. Apply explicit CLI overrides (only when the user actually passed them)
    if args.model_id is not None:
        cfg["model_id"] = args.model_id
    if args.adapter_id is not None:
        cfg["adapter_id"] = args.adapter_id
    if args.lora_id is not None:
        cfg["lora_id"] = args.lora_id
    if args.lora_scale is not None:
        cfg["lora_scale"] = args.lora_scale
    if args.steps is not None:
        cfg["num_inference_steps"] = args.steps
    if args.guidance_scale is not None:
        cfg["guidance_scale"] = args.guidance_scale

    # 4. Trigger-word check — prepend automatically if missing from the prompt
    trigger: Optional[str] = cfg.get("trigger_word") or None
    prompt: str = args.prompt
    if trigger and trigger.lower() not in prompt.lower():
        print(
            f"⚠️  Trigger word {trigger!r} not found in prompt — "
            f"prepending it automatically."
        )
        prompt = f"{trigger} {prompt}"
    cfg["_effective_prompt"] = prompt  # resolved prompt passed to the pipeline

    # 5. Resolve output path and ensure the directory exists
    if args.output:
        output_path = args.output
    else:
        # Auto-generate a timestamped filename so runs never overwrite each other.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(cfg["output_dir"], f"{timestamp}.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    return cfg, output_path


def print_config(cfg: dict, args: argparse.Namespace, output_path: str) -> None:
    """Print a human-readable summary of the effective configuration.

    Args:
        cfg: Fully resolved configuration dict (from build_config).
        args: Parsed argument namespace (for prompt / negative_prompt).
        output_path: Resolved output file path.
    """
    print("── Effective configuration ──────────────────────────────────────")
    if cfg.get("description"):
        print(f"  description       : {cfg['description']}")
    print(f"  pipeline_type     : {cfg.get('pipeline_type', 'sd')}")
    print(f"  model_id          : {cfg['model_id']}")
    print(f"  adapter_id        : {cfg['adapter_id'] or '(none)'}")
    print(f"  lora_id           : {cfg['lora_id'] or '(none)'}")
    print(f"  lora_scale        : {cfg['lora_scale']}")
    if cfg.get("trigger_word"):
        print(f"  trigger_word      : {cfg['trigger_word']!r}")
    print(f"  steps             : {cfg['num_inference_steps']}")
    print(f"  guidance_scale    : {cfg['guidance_scale']}")
    print(f"  width x height    : {cfg.get('width', 512)} x {cfg.get('height', 512)}")
    print(f"  cpu_offload       : {cfg['sequential_cpu_offload']}")
    print(f"  prompt            : {cfg['_effective_prompt']!r}")
    if cfg["_effective_prompt"] != args.prompt:
        print(f"  prompt (original) : {args.prompt!r}")
    print(f"  negative_prompt   : {args.negative_prompt!r}")
    print(f"  output            : {output_path}")
    print("─────────────────────────────────────────────────────────────────")
