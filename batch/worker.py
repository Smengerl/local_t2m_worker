"""
Batch worker — processes the FIFO queue sequentially.

The worker runs as an asyncio coroutine.  ``process_job()`` itself is
blocking (PyTorch inference) and always runs in a ``ThreadPoolExecutor`` via
``asyncio.run_in_executor`` so the event loop stays free.

Idle behaviour is controlled by the ``keep_alive`` parameter:
  * ``keep_alive=True``  — wait for a job-ready ``asyncio.Event`` when the
                           queue is empty; run forever until cancelled.
                           Used by the FastAPI server (event supplied by
                           ``batch.notify``).
  * ``keep_alive=False`` — exit once all current pending jobs are done.
                           Default for the CLI entry point.

Cancellation of the *current job* (not the whole worker) is requested via
``request_cancel()``, which sets a thread-safe flag that the ``_progress``
callback inside ``process_job`` checks between denoising steps.

Usage (CLI):
    python -m batch.worker               # process pending jobs, then exit
    python -m batch.worker --keep-alive  # stay alive, wait for new jobs
"""

import argparse
import asyncio
import gc
import logging
import os
import re
import signal
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Make sure the project root is on sys.path when run as a module ────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch.queue import mark_done, mark_failed, mark_running, next_pending, update_job, append_log
from batch import notify
from generate import generate_image
from pipeline_config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker")

# Maximum seconds to wait for the current inference thread to finish after a
# cancellation or SIGTERM before giving up.  Matches the server-side lifespan
# shutdown timeout so behaviour is identical in both modes.
_SHUTDOWN_TIMEOUT_S = 10

# Force tqdm to always write progress bars, even when stdout/stderr is not a
# TTY.  Must be set before any tqdm import so the global default is picked up.
os.environ.setdefault("TQDM_DISABLE", "0")


class _CancellationError(Exception):
    """Raised to cancel the current job without stopping the worker."""


# ── Cancel flag (thread-safe) ─────────────────────────────────────────────────
# Set by request_cancel() (called from the HTTP cancel endpoint or the CLI
# signal handler) and cleared at the start of every job.
# The _progress callback inside process_job checks it between denoising steps.
_cancel_event = threading.Event()


def request_cancel() -> None:
    """Request cancellation of the currently running job (thread-safe)."""
    _cancel_event.set()


# ── Worker state (readable by /api/health) ────────────────────────────────────
# These module-level variables are written only by the worker coroutine/thread
# and read by the health endpoint.  All are set before/after process_job so
# no additional locking is required beyond the GIL.

# Key of the currently loaded pipeline, or None when the cache is empty.
# Matches the key format used in pipeline_cache: (config_hash, ...).
# Exposed as a human-readable model repo string for the health endpoint.
_cached_model: str | None = None

# Job ID currently being executed, or None when idle.
_current_job_id: str | None = None


# ── Logging helpers ───────────────────────────────────────────────────────────

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

    def __init__(self, job_id: str, real_stderr: Any) -> None:
        self.job_id = job_id
        self.real_stderr = real_stderr
        self._buf = ""  # accumulate until newline or CR

    def write(self, text: str) -> int:
        # Always pass through to the real stderr (terminal / log file).
        self.real_stderr.write(text)
        self.real_stderr.flush()

        # Strip ANSI escape sequences.
        text = self._ANSI_RE.sub("", text)

        # \r: tqdm rewrites the same line — discard buffer, log only on \n.
        for ch in text:
            if ch == "\r":
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
        # Pretend to be a TTY so tqdm/huggingface_hub don't suppress progress bars.
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real_stderr, name)


# ── Core job execution ────────────────────────────────────────────────────────

def process_job(
    job: dict[str, Any],
    pipeline_cache: dict[tuple, Any],
) -> str:
    """Run a single job. Returns the result path. Raises on failure.

    Blocking — always call from a ThreadPoolExecutor, never from the event loop.
    """
    global _current_job_id

    log.info("Starting job %s  prompt=%r", job["id"][:8], job["prompt"])
    mark_running(job["id"], worker_pid=os.getpid())
    _current_job_id = job["id"]
    _cancel_event.clear()  # reset any leftover cancel request from a previous job

    # ── attach per-job log handler to the root logger ─────────────────────────
    job_handler = _JobLogHandler(job["id"])
    job_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(job_handler)

    # ── redirect stderr so tqdm/hf-hub progress bars appear in web logs ───────
    real_stderr = sys.stderr
    sys.stderr = _StderrCapture(job["id"], real_stderr)

    try:
        cfg = PipelineConfig.from_dict(job["pipeline_config"])

        output_path = job.get("output") or str(
            Path(cfg.output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

        job_id = job["id"]

        def _progress(step: int, total: int) -> None:
            if _cancel_event.is_set():
                raise _CancellationError(f"Cancelled by user request (job {job_id[:8]})")
            update_job(job_id, progress_step=step, progress_total=total)

        generate_image(
            cfg, output_path,
            job["prompt"], job.get("negative_prompt") or "",
            pipeline_cache=pipeline_cache,
            progress_callback=_progress,
        )
        log.info("Job %s done → %s", job["id"][:8], output_path)
        return output_path
    finally:
        _current_job_id = None
        sys.stderr = real_stderr
        root_logger.removeHandler(job_handler)


def _release_pipeline_cache(pipeline_cache: dict[tuple, Any]) -> None:
    """Explicitly release cached pipeline objects and free GPU/MPS memory.

    On Apple Silicon (MPS) PyTorch pipelines crash the interpreter during normal
    GC at shutdown if the MPS backend has already started tearing down.  Clearing
    the cache here — before the interpreter begins to exit — avoids that race.
    """
    global _cached_model
    pipeline_cache.clear()
    _cached_model = None
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # best-effort; never crash the worker on cleanup


def _finish_job(
    job: dict[str, Any],
    result_path: str | None,
    exc: BaseException | None,
    pipeline_cache: dict[tuple, Any],
) -> None:
    """Centralised post-job bookkeeping.

    Called after process_job() returns or raises.  Exactly one of
    ``result_path`` (success) or ``exc`` (failure/cancel) will be non-None.
    """
    if result_path is not None:
        mark_done(job["id"], result_path)
        return

    assert exc is not None
    if isinstance(exc, _CancellationError):
        log.info("Job %s cancelled.", job["id"][:8])
        mark_failed(job["id"], str(exc))
        _cancel_event.clear()
    else:
        log.error("Job %s FAILED: %s", job["id"][:8], exc)
        mark_failed(job["id"], traceback.format_exc())
    # Pipeline state may be inconsistent after a cancel or unexpected error.
    _release_pipeline_cache(pipeline_cache)


# ── Unified worker loop ───────────────────────────────────────────────────────

async def run_worker_async(*, keep_alive: bool = True) -> None:
    """Unified async worker loop — used by both the CLI and the FastAPI server.

    Args:
        keep_alive: When ``True`` (default) the worker registers itself with
                    ``batch.notify`` and waits for new jobs when the queue is
                    empty.  Runs until the task is cancelled (Ctrl-C / SIGTERM
                    / server shutdown).
                    When ``False`` the worker exits as soon as no more pending
                    jobs remain — useful for one-shot CLI use.
    """
    loop = asyncio.get_running_loop()
    pipeline_cache: dict[tuple, Any] = {}
    global _cached_model

    # Register with the notification module so POST /api/jobs (and the CLI
    # equivalent) can wake this loop without polling.
    event: asyncio.Event | None = notify.init() if keep_alive else None

    # Pre-set the event if there are already pending jobs in the queue at
    # startup.  This covers two cases:
    #   1. Jobs were added before the server started (no API request, so
    #      notify() was never called for them).
    #   2. Jobs were submitted via POST /api/jobs in the brief window between
    #      the server accepting connections and this worker task being
    #      scheduled for the first time — in that window notify() runs from
    #      a FastAPI thread but notify.init() has not been called yet, so
    #      _event is still None and the call is silently dropped.
    if event is not None and next_pending() is not None:
        event.set()
        log.info("Pre-existing pending jobs found — worker starting immediately.")

    log.info("Worker started (keep_alive=%s).", keep_alive)

    try:
        while True:
            job = next_pending()
            if job is None:
                if not keep_alive:
                    log.info("No pending jobs left — exiting.")
                    break
                # Double-check after clearing to close the TOCTOU window.
                assert event is not None
                event.clear()
                if next_pending() is None:
                    # Use wait_for with a timeout as a safety-net poll.
                    # notify() is the primary wake-up mechanism, but it can be
                    # silently dropped in a narrow startup race window: if a
                    # POST /api/jobs request arrives between the server
                    # accepting connections and notify.init() being called by
                    # this coroutine, _event is still None so notify() is a
                    # no-op.  The timeout (5 s) ensures the worker recovers
                    # automatically without requiring a server restart.
                    try:
                        await asyncio.wait_for(event.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass  # normal — just re-check the queue
                continue

            result_path: str | None = None
            caught: BaseException | None = None

            # Check if this job requires a different model than the one currently cached.
            # If so, clear the cache beforehand to free up memory for the new model.
            # This is critical on devices with limited RAM (like MacBook Air).
            job_config = job.get("pipeline_config", {})
            target_model = job_config.get("model", {}).get("repo")
            if target_model and _cached_model and target_model != _cached_model:
                log.info("Model change detected (%s -> %s). Clearing cache...", 
                         _cached_model, target_model)
                _release_pipeline_cache(pipeline_cache)

            try:
                fut = loop.run_in_executor(None, process_job, job, pipeline_cache)
                result_path = await fut
            except asyncio.CancelledError:
                # Der Worker-Task selbst wurde abgebrochen (z.B. durch Server-Shutdown).
                # Wir müssen den Hintergrund-Thread benachrichtigen und auf ihn warten,
                # um verwaiste Threads und Abstürze beim Beenden des Interpreters zu vermeiden.
                request_cancel()
                log.info("Worker cancelled — waiting for current job thread...")
                try:
                    # Dem Thread Zeit geben, das Abbruch-Flag zu sehen und sich sauber zu beenden.
                    await asyncio.wait_for(fut, timeout=_SHUTDOWN_TIMEOUT_S)
                except (asyncio.TimeoutError, Exception):
                    # Falls es zu lange dauert, fahren wir trotzdem mit dem Shutdown fort.
                    pass

                _finish_job(job, None, _CancellationError("Worker shutdown"), pipeline_cache)
                raise
            except (_CancellationError, Exception) as exc:
                caught = exc
            _finish_job(job, result_path, caught, pipeline_cache)

            # Update cached-model indicator from the pipeline_cache key.
            # The cache maps (model_repo, ...) tuples → pipeline objects.
            # After _finish_job the cache may be empty (on error) or still
            # populated (on success); reflect the current state.
            _cached_model = next(
                (k[0] for k in pipeline_cache if isinstance(k, tuple) and k),
                None,
            )

    except asyncio.CancelledError:
        log.info("Worker task cancelled — shutting down.")
    finally:
        _release_pipeline_cache(pipeline_cache)
        notify.reset()
        log.info("Worker shut down cleanly.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch worker — processes the image-generation queue.",
    )
    parser.add_argument(
        "--keep-alive", action="store_true", default=False,
        help=(
            "Stay alive after the queue is empty and wait for new jobs "
            "instead of exiting.  Without this flag the worker exits as "
            "soon as all current pending jobs are done."
        ),
    )
    args = parser.parse_args()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None

        # Translate SIGTERM into task cancellation so the worker shuts down
        # cleanly (pipeline cache released, job marked failed) instead of
        # being killed mid-inference.
        loop.add_signal_handler(signal.SIGTERM, task.cancel)

        try:
            await run_worker_async(keep_alive=args.keep_alive)
        except asyncio.CancelledError:
            pass  # normal shutdown path — already handled inside run_worker_async

    asyncio.run(_run())


if __name__ == "__main__":
    main()
