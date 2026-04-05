"""
Batch worker — processes the FIFO queue sequentially.

The worker polls queue.jsonl for pending jobs, runs them one by one using the
same generate.py logic, and updates each job's status in the queue file.

The pipeline is loaded once per unique (config + all overrides) combination.
When the next job requires the same model, the already-loaded pipeline is
reused; otherwise it is replaced.  This avoids the expensive reload on
back-to-back jobs that share the same config.

Usage:
    python -m batch.worker               # poll every 5 s (default)
    python -m batch.worker --poll 10     # poll every 10 s
    python -m batch.worker --once        # process all current pending jobs, then exit
"""

import argparse
import logging
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Make sure the project root is on sys.path when run as a module ────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch.queue import mark_done, mark_failed, mark_running, next_pending, update_job, append_log
from cli import build_config, load_config
from pipelines import create_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker")


class _JobLogHandler(logging.Handler):
    """Logging handler that appends formatted records to a job's log_lines."""

    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            append_log(self.job_id, self.format(record))
        except Exception:
            pass  # never crash the worker due to logging


class _StderrCapture:
    """File-like object that replaces sys.stderr during job execution.

    tqdm and huggingface_hub write download/generation progress bars directly
    to stderr using carriage-return (\\r) overwrite sequences — they never go
    through Python's logging system, so _JobLogHandler alone misses them.

    This class intercepts those writes, strips ANSI escape codes and CR tricks,
    and forwards meaningful lines to append_log so they appear in the web UI.
    """

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def __init__(self, job_id: str, real_stderr) -> None:
        self.job_id = job_id
        self.real_stderr = real_stderr
        self._buf = ""  # accumulate until newline or CR

    def write(self, text: str) -> int:
        # Always pass through to the real stderr (terminal / log file).
        self.real_stderr.write(text)
        self.real_stderr.flush()

        # Strip ANSI escape sequences.
        text = self._ANSI_RE.sub("", text)

        # Process character by character so we handle \r and \n correctly.
        for ch in text:
            if ch == "\r":
                # CR: tqdm "overwrites" the current line — we treat the
                # accumulated buffer as a complete (intermediate) line and
                # reset it so the next write starts fresh.
                line = self._buf.strip()
                if line:
                    try:
                        append_log(self.job_id, line)
                    except Exception:
                        pass
                self._buf = ""
            elif ch == "\n":
                line = self._buf.strip()
                if line:
                    try:
                        append_log(self.job_id, line)
                    except Exception:
                        pass
                self._buf = ""
            else:
                self._buf += ch

        return len(text)

    def flush(self) -> None:
        # Flush any incomplete line that didn't end with \n or \r.
        line = self._buf.strip()
        if line:
            try:
                append_log(self.job_id, line)
            except Exception:
                pass
            self._buf = ""
        self.real_stderr.flush()

    def fileno(self) -> int:
        return self.real_stderr.fileno()

    def isatty(self) -> bool:
        return False

    # Delegate everything else (e.g. .encoding, .errors) to the real stderr.
    def __getattr__(self, name: str):
        return getattr(self.real_stderr, name)

# ─────────────────────────────────────────────────────────────────────────────


def _cache_key(job: dict[str, Any]) -> tuple:
    """Key that identifies a unique pipeline configuration."""
    return (
        job["config"],
        job.get("model_id"),
        job.get("adapter_id"),
        job.get("lora_id"),
        job.get("lora_scale"),
    )


def _build_fake_args(job: dict[str, Any]):
    """Build an argparse.Namespace from a queue job (mirrors parse_args output)."""
    import argparse
    return argparse.Namespace(
        config=job["config"],
        prompt=job["prompt"],
        negative_prompt=job.get("negative_prompt") or "",
        output=job.get("output"),
        output_dir=job.get("output_dir"),
        cache_dir=job.get("cache_dir"),
        model_id=job.get("model_id"),
        adapter_id=job.get("adapter_id"),
        lora_id=job.get("lora_id"),
        lora_scale=job.get("lora_scale"),
        steps=job.get("steps"),
        guidance_scale=job.get("guidance_scale"),
    )


def process_job(
    job: dict[str, Any],
    pipeline_cache: dict[tuple, Any],
) -> str:
    """Run a single job. Returns the result path. Raises on failure."""
    log.info("Starting job %s  prompt=%r  config=%s", job["id"][:8], job["prompt"], job["config"])
    mark_running(job["id"])

    # ── attach per-job log handler to the root logger (captures all libs) ──
    job_handler = _JobLogHandler(job["id"])
    job_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(job_handler)

    # ── redirect stderr so tqdm/hf-hub progress bars appear in web logs ──────
    real_stderr = sys.stderr
    sys.stderr = _StderrCapture(job["id"], real_stderr)

    try:
        args = _build_fake_args(job)
        cfg, output_path = build_config(args)

        key = _cache_key(job)
        if key not in pipeline_cache:
            log.info("Loading pipeline for key %s ...", key)
            pipeline_cache.clear()           # free memory before loading new model
            pipeline_cache[key] = create_pipeline(cfg)
            log.info("Pipeline loaded.")
        else:
            log.info("Reusing cached pipeline.")

        pipeline = pipeline_cache[key]

        # ── progress callback: writes live step counter into queue.jsonl ──────
        job_id = job["id"]

        def _progress(step: int, total: int) -> None:
            update_job(job_id, progress_step=step, progress_total=total)

        image = pipeline.generate(
            prompt=cfg["_effective_prompt"],
            negative_prompt=args.negative_prompt,
            progress_callback=_progress,
        )
        image.save(output_path)
        log.info("Job %s done → %s", job["id"][:8], output_path)
        return output_path
    finally:
        sys.stderr = real_stderr
        root_logger.removeHandler(job_handler)


def run_worker(poll_interval: int = 5, run_once: bool = False) -> None:
    """Main worker loop."""
    log.info("Worker started (poll_interval=%ds, run_once=%s)", poll_interval, run_once)
    pipeline_cache: dict[tuple, Any] = {}

    while True:
        job = next_pending()
        if job is None:
            if run_once:
                log.info("No pending jobs left — exiting (--once).")
                break
            time.sleep(poll_interval)
            continue

        try:
            result_path = process_job(job, pipeline_cache)
            mark_done(job["id"], result_path)
        except Exception as exc:
            err = traceback.format_exc()
            log.error("Job %s FAILED: %s", job["id"][:8], exc)
            mark_failed(job["id"], err)
            pipeline_cache.clear()   # discard potentially broken pipeline state

        if run_once:
            # keep consuming until queue is empty
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch worker — processes the image-generation queue.",
    )
    parser.add_argument(
        "--poll", type=int, default=5, metavar="SECONDS",
        help="Seconds to wait between queue polls when idle (default: 5).",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Process all current pending jobs and exit instead of looping forever.",
    )
    args = parser.parse_args()
    run_worker(poll_interval=args.poll, run_once=args.once)


if __name__ == "__main__":
    main()
