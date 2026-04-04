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
    "lora_weights": None,
    "lora_scale": 0.9,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "output_dir": "outputs",
    "cache_dir": "models",
    # Offload model submodules to CPU between steps to save GPU/MPS memory.
    # Slower, but necessary for SDXL on machines with ≤16 GB unified memory.
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
        help="Hugging Face model ID or local path. Overrides config.",
    )
    parser.add_argument(
        "--lora-weights",
        metavar="REPO_ID",
        help="Hugging Face LoRA repo ID or local path. Overrides config.",
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

    # 2. Apply explicit CLI overrides (only when the user actually passed them)
    if args.model_id is not None:
        cfg["model_id"] = args.model_id
    if args.lora_weights is not None:
        cfg["lora_weights"] = args.lora_weights
    if args.lora_scale is not None:
        cfg["lora_scale"] = args.lora_scale
    if args.steps is not None:
        cfg["num_inference_steps"] = args.steps
    if args.guidance_scale is not None:
        cfg["guidance_scale"] = args.guidance_scale

    # 3. Resolve output path and ensure the directory exists
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
    print(f"  model_id          : {cfg['model_id']}")
    print(f"  lora_weights      : {cfg['lora_weights'] or '(none)'}")
    print(f"  lora_scale        : {cfg['lora_scale']}")
    print(f"  steps             : {cfg['num_inference_steps']}")
    print(f"  guidance_scale    : {cfg['guidance_scale']}")
    print(f"  cpu_offload       : {cfg['sequential_cpu_offload']}")
    print(f"  prompt            : {args.prompt!r}")
    print(f"  negative_prompt   : {args.negative_prompt!r}")
    print(f"  output            : {output_path}")
    print("─────────────────────────────────────────────────────────────────")
