"""
CLI tool for adding jobs to the batch queue.

Accepts exactly the same flags as generate.py / run.sh so users can
switch between one-shot and queued execution by swapping the command.

Usage:
    python -m batch.enqueue -c configs/sdxl_graffiti_lora.json "a dragon"
    python -m batch.enqueue --help
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch.queue import enqueue, stats
from cli import build_config, parse_args


def main() -> None:
    args = parse_args()
    cfg, output_path, effective_prompt, negative_prompt = build_config(args)

    job = enqueue(
        cfg=cfg,
        prompt=effective_prompt,
        negative_prompt=negative_prompt,
        output=args.output,
    )

    s = stats()
    print(f"✅ Job enqueued  id={job['id'][:8]}…")
    print(f"   config  : {job['pipeline_config']['pipeline_type']}  {job['pipeline_config']['model_id']}")
    print(f"   prompt  : {job['prompt']!r}")
    print(f"   Queue   : {s['pending']} pending  {s['running']} running  "
          f"{s['done']} done  {s['failed']} failed")


if __name__ == "__main__":
    main()
