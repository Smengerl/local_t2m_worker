"""
CLI tool for adding jobs to the batch queue.

Accepts exactly the same flags as generate.py / run.sh so users can
switch between one-shot and queued execution by swapping the command.

Usage:
    python -m batch.enqueue -c configs/sdxl_graffiti_lora.json "a dragon"
    python -m batch.enqueue --help
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch.queue import enqueue, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enqueue an image-generation job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m batch.enqueue 'a sunset over mountains'\n"
            "  python -m batch.enqueue -c configs/sdxl_graffiti_lora.json 'graffiti dragon'\n"
            "  python -m batch.enqueue --steps 50 --guidance-scale 8 'a cat'\n"
        ),
    )
    parser.add_argument("prompt", metavar="PROMPT",
                        help="Text prompt for the image to generate.")
    parser.add_argument("--config", "-c", metavar="FILE",
                        default="configs/sd15_default.json",
                        help="JSON config file (default: configs/sd15_default.json).")
    parser.add_argument("--negative-prompt", "-n", default="", metavar="TEXT",
                        help="Negative prompt.")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Output PNG path (default: auto-timestamped).")
    parser.add_argument("--model-id", metavar="REPO_ID",
                        help="Override model_id from config.")
    parser.add_argument("--adapter-id", metavar="REPO_ID",
                        help="Override adapter_id from config.")
    parser.add_argument("--lora-id", metavar="REPO_ID",
                        help="Override lora_id from config.")
    parser.add_argument("--lora-scale", type=float, metavar="FLOAT",
                        help="Override lora_scale from config.")
    parser.add_argument("--steps", type=int, metavar="N",
                        help="Override num_inference_steps from config.")
    parser.add_argument("--guidance-scale", type=float, metavar="FLOAT",
                        help="Override guidance_scale from config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    job = enqueue(
        config=args.config,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output=args.output,
        model_id=args.model_id,
        adapter_id=args.adapter_id,
        lora_id=args.lora_id,
        lora_scale=args.lora_scale,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    s = stats()
    print(f"✅ Job enqueued  id={job['id'][:8]}…")
    print(f"   config  : {job['config']}")
    print(f"   prompt  : {job['prompt']!r}")
    print(f"   Queue   : {s['pending']} pending  {s['running']} running  "
          f"{s['done']} done  {s['failed']} failed")


if __name__ == "__main__":
    main()
