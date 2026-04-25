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
import os
import re
import signal
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

from batch.queue import mark_done, mark_failed, mark_running, next_pending, update_job, append_log, set_worker_pid
from generate import generate_image
from pipeline_config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker")

# Force tqdm to always write progress bars, even when stdout/stderr is not a
# TTY.  Must be set before any tqdm import so the global default is picked up.
os.environ.setdefault("TQDM_DISABLE", "0")


class _CancellationError(Exception):
    """Raised by the SIGTERM handler to cancel the current job without killing the worker."""


# Tracks the job ID currently being processed so the SIGTERM handler knows
# whether to cancel a job or shut down the worker.
_running_job_id: Optional[str] = None


def _install_sigterm_handler() -> None:
    """Install a SIGTERM handler that cancels the active job without stopping the worker loop.

    If a job is currently running, SIGTERM raises _CancellationError which unwinds
    generate_image() and is caught by the worker loop — the loop then continues with
    the next pending job.  If the worker is idle, SIGTERM exits cleanly via sys.exit(0).
    """
    def _handler(signum: int, frame: Any) -> None:  # noqa: ANN001
        if _running_job_id is not None:
            log.info("SIGTERM received — cancelling job %s", _running_job_id[:8])
            raise _CancellationError(f"Cancelled via SIGTERM (job {_running_job_id[:8]})")
        # No job in progress — honour the signal and exit the worker.
        log.info("SIGTERM received while idle — worker exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handler)


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
        # \r (carriage-return): tqdm rewrites the same line — discard the
        # intermediate buffer so only the *final* state of the line is logged.
        # \n: flush the buffer as a completed log line.
        for ch in text:
            if ch == "\r":
                # Discard accumulated text — next write will overwrite it.
                # We only log on \n so the web UI sees the last/final value.
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
        # Pretend to be a TTY so tqdm and huggingface_hub don't suppress their
        # progress bars.  Without this, tqdm detects a non-interactive stream
        # and either disables or heavily throttles output.
        return True

    # Delegate everything else (e.g. .encoding, .errors) to the real stderr.
    def __getattr__(self, name: str):
        return getattr(self.real_stderr, name)

# ─────────────────────────────────────────────────────────────────────────────


def process_job(
    job: dict[str, Any],
    pipeline_cache: dict[tuple, Any],
) -> str:
    """Run a single job. Returns the result path. Raises on failure."""
    global _running_job_id

    log.info("Starting job %s  prompt=%r", job["id"][:8], job["prompt"])
    mark_running(job["id"])
    set_worker_pid(job["id"], os.getpid())
    _running_job_id = job["id"]

    # ── attach per-job log handler to the root logger (captures all libs) ──
    job_handler = _JobLogHandler(job["id"])
    job_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(job_handler)

    # ── redirect stderr so tqdm/hf-hub progress bars appear in web logs ──────
    real_stderr = sys.stderr
    sys.stderr = _StderrCapture(job["id"], real_stderr)

    try:
        cfg = PipelineConfig.from_dict(job["pipeline_config"])

        # Resolve output path: use explicit path from job or auto-generate.
        if job.get("output"):
            output_path = job["output"]
        else:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(Path(cfg.output_dir) / f"{ts}.png")

        effective_prompt = job["prompt"]
        negative_prompt = job.get("negative_prompt") or ""

        job_id = job["id"]

        def _progress(step: int, total: int) -> None:
            update_job(job_id, progress_step=step, progress_total=total)

        generate_image(cfg, output_path, effective_prompt, negative_prompt, pipeline_cache=pipeline_cache, progress_callback=_progress)
        log.info("Job %s done → %s", job["id"][:8], output_path)
        return output_path
    finally:
        _running_job_id = None
        sys.stderr = real_stderr
        root_logger.removeHandler(job_handler)


def run_worker(poll_interval: int = 5, run_once: bool = False) -> None:
    """Main worker loop."""
    log.info("Worker started (poll_interval=%ds, run_once=%s)", poll_interval, run_once)

    # Write PID file so external tools (e.g. run.sh) can check if we're alive.
    pid_file = _ROOT / "batch" / "worker.pid"
    pid_file.write_text(str(os.getpid()))
    try:
        _run_worker_loop(poll_interval=poll_interval, run_once=run_once)
    finally:
        pid_file.unlink(missing_ok=True)


def _run_worker_loop(poll_interval: int = 5, run_once: bool = False) -> None:
    """Inner worker loop (separated so the PID file teardown is always clean)."""
    _install_sigterm_handler()
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
        except _CancellationError as exc:
            # Job was cancelled via SIGTERM from the web UI / cancel tool.
            # The API already called mark_failed; call it again to record the
            # cancellation reason in the error field (idempotent).
            log.info("Job %s cancelled.", job["id"][:8])
            mark_failed(job["id"], str(exc))
            pipeline_cache.clear()   # pipeline state may be inconsistent
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
