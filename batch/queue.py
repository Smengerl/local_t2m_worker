"""
Persistent FIFO job queue backed by a JSONL file.

Each line in the file is one JSON object representing a job:
  {
    "id":               str,        # UUID4
    "status":           str,        # "pending" | "running" | "done" | "failed"
    "added_at":         str,        # ISO-8601 timestamp
    "started_at":       str|null,
    "finished_at":      str|null,
    "pipeline_config":  dict,       # serialised PipelineConfig (all pipeline params)
    "prompt":           str,
    "negative_prompt":  str,
    "output":           str|null,   # explicit output path, or null (auto-generated)
    "progress_step":    int,        # current denoising step (updated live by worker)
    "progress_total":   int,        # total denoising steps (0 = not started / unknown)
    "worker_pid":       int|null,   # PID of the worker process while running, else null
    "result_path":      str|null,   # set by worker when done
    "error":            str|null    # set by worker when failed
  }

Thread-/process-safety is provided by filelock so the web server and the
worker can access the queue file concurrently without corruption.
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline_config import PipelineConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_FILE = _PROJECT_ROOT / "queue.jsonl"
_LOCK_FILE  = _PROJECT_ROOT / "queue.jsonl.lock"
# ─────────────────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock() -> FileLock:
    return FileLock(str(_LOCK_FILE), timeout=10)


# ── Read / write helpers ──────────────────────────────────────────────────────

def _read_all() -> list[dict[str, Any]]:
    """Read all jobs from the queue file (no lock — caller must hold it)."""
    if not QUEUE_FILE.exists():
        return []
    jobs = []
    for line in QUEUE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            jobs.append(json.loads(line))
    return jobs


def _write_all(jobs: list[dict[str, Any]]) -> None:
    """Rewrite the entire queue file (no lock — caller must hold it)."""
    QUEUE_FILE.write_text(
        "\n".join(json.dumps(j, ensure_ascii=False) for j in jobs) + "\n",
        encoding="utf-8",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def enqueue(
    *,
    cfg: PipelineConfig,
    prompt: str,
    negative_prompt: str = "",
    output: Optional[str] = None,
) -> dict[str, Any]:
    """Append a new pending job to the queue and return it."""
    job: dict[str, Any] = {
        "id":              str(uuid.uuid4()),
        "status":          "pending",
        "added_at":        _now(),
        "started_at":      None,
        "finished_at":     None,
        "pipeline_config": cfg.to_dict(),
        "prompt":          prompt,
        "negative_prompt": negative_prompt,
        "output":          output,
        "progress_step":   0,
        "progress_total":  0,
        "log_lines":       [],
        "worker_pid":      None,
        "result_path":     None,
        "error":           None,
    }
    with _lock():
        jobs = _read_all()
        jobs.append(job)
        _write_all(jobs)
    return job


def list_jobs() -> list[dict[str, Any]]:
    """Return all jobs (oldest first)."""
    with _lock():
        return _read_all()


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Return a single job by ID, or None if not found."""
    with _lock():
        for job in _read_all():
            if job["id"] == job_id:
                return job
    return None


def next_pending() -> Optional[dict[str, Any]]:
    """Return the oldest pending job without modifying the queue."""
    with _lock():
        for job in _read_all():
            if job["status"] == "pending":
                return job
    return None

def reorder_pending(job_ids: list[str]) -> None:
    """Change the order of pending jobs based on a list of IDs.

    Jobs not in 'pending' status are left at their current positions in the file.
    Pending jobs missing from 'job_ids' are appended after the reordered ones.
    """
    with _lock():
        all_jobs = _read_all()

        # Extract all pending jobs and map them by ID
        pending_map = {j["id"]: j for j in all_jobs if j["status"] == "pending"}

        # Build the new sequence of pending jobs
        new_pending_list = []
        for jid in job_ids:
            if jid in pending_map:
                new_pending_list.append(pending_map.pop(jid))

        # Any pending jobs NOT in the input list go to the end of the pending group
        new_pending_list.extend(pending_map.values())

        # Reassemble all jobs: replace old pending slots with the new sequence
        final_jobs = []
        pending_iter = iter(new_pending_list)
        for j in all_jobs:
            if j["status"] == "pending":
                final_jobs.append(next(pending_iter))
            else:
                final_jobs.append(j)

        _write_all(final_jobs)

        
def update_job(job_id: str, **fields: Any) -> Optional[dict[str, Any]]:
    """Update fields on a job in-place and return the updated job."""
    with _lock():
        jobs = _read_all()
        for job in jobs:
            if job["id"] == job_id:
                job.update(fields)
                _write_all(jobs)
                return job
    return None


def append_log(job_id: str, line: str, max_lines: int = 300) -> None:
    """Append a log line to a job's log_lines list (capped at max_lines).

    Note: this rewrites the entire queue file on every call.  It is called
    frequently during job execution (every denoising step + tqdm lines), so
    keep max_lines small to bound write amplification.
    """
    with _lock():
        jobs = _read_all()
        for job in jobs:
            if job["id"] == job_id:
                lines: list = job.setdefault("log_lines", [])
                lines.append(line)
                if len(lines) > max_lines:
                    del lines[:len(lines) - max_lines]
                _write_all(jobs)
                return


def mark_running(job_id: str, worker_pid: Optional[int] = None) -> Optional[dict[str, Any]]:
    return update_job(job_id, status="running", started_at=_now(),
                      progress_step=0, progress_total=0, log_lines=[],
                      worker_pid=worker_pid)


def mark_done(job_id: str, result_path: str) -> Optional[dict[str, Any]]:
    return update_job(job_id, status="done", finished_at=_now(),
                      result_path=result_path, worker_pid=None)


def mark_failed(job_id: str, error: str) -> Optional[dict[str, Any]]:
    return update_job(job_id, status="failed", finished_at=_now(), error=error,
                      progress_step=0, progress_total=0, worker_pid=None)


def set_worker_pid(job_id: str, pid: int) -> Optional[dict[str, Any]]:
    """Store the PID of the worker process handling this job."""
    return update_job(job_id, worker_pid=pid)


def delete_job(job_id: str) -> bool:
    """Remove a job from the queue. Returns True if found and deleted."""
    with _lock():
        jobs = _read_all()
        new_jobs = [j for j in jobs if j["id"] != job_id]
        if len(new_jobs) == len(jobs):
            return False
        _write_all(new_jobs)
    return True


def clear_finished() -> int:
    """Remove all done/failed jobs. Returns number of removed jobs."""
    with _lock():
        jobs = _read_all()
        keep = [j for j in jobs if j["status"] in ("pending", "running")]
        removed = len(jobs) - len(keep)
        _write_all(keep)
    return removed


def stats() -> dict[str, int]:
    """Return counts per status."""
    counts: dict[str, int] = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    with _lock():
        for job in _read_all():
            s = job.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1
    return counts
