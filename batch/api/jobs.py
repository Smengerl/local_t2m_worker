"""
REST endpoints for job management.

Routes:
  GET    /api/jobs                  – list all jobs
  GET    /api/jobs/{id}             – single job
  POST   /api/jobs                  – enqueue new job
  DELETE /api/jobs/{id}             – delete pending/done/failed job
  POST   /api/jobs/{id}/retry       – re-queue done/failed job
  POST   /api/jobs/{id}/cancel      – cancel running/pending job
  POST   /api/clear-finished        – remove all done/failed jobs
  GET    /api/stats                 – job counts per status
"""

import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch.queue import (
    clear_finished,
    delete_job,
    enqueue,
    get_job,
    list_jobs,
    mark_failed,
    append_log,
    stats,
    update_job,
)
from batch import notify
from batch import worker as _worker
from pipeline_config import PipelineConfig

router = APIRouter()


def _is_pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* is currently running.

    Uses os.kill(pid, 0) which sends no signal but raises OSError if the
    process does not exist or is not accessible.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we cannot signal it — treat as alive.
        return True


def _heal_stale_running_jobs() -> None:
    """Mark any 'running' job as failed if its worker process is no longer alive.

    Called on every GET /api/jobs so the UI always reflects reality without
    requiring a manual cancel action from the user.
    """
    for job in list_jobs():
        if job.get("status") != "running":
            continue
        pid = job.get("worker_pid")
        if pid is None:
            # No PID recorded — job may have just been picked up by the worker;
            # give it a grace period by leaving it alone.
            continue
        if not _is_pid_alive(pid):
            msg = f"Worker process (PID {pid}) no longer exists — marked as failed by server."
            append_log(job["id"], f"[server] {msg}")
            mark_failed(job["id"], msg)


class EnqueueRequest(BaseModel):
    """Request body for POST /api/jobs.

    ``config`` is the only required non-prompt field; all other fields are
    per-request overrides that shadow the values from the config file.
    Override field names mirror the config file key paths (section.field)
    flattened with an underscore — matching the CLI flag names exactly
    (e.g. ``model_repo`` ↔ ``--model-repo`` ↔ ``model.repo`` in the config).
    """

    config: str = "configs/sd15_default.json"
    prompt: str
    negative_prompt: str = ""
    output: Optional[str] = None
    # Model overrides — mirror model.* config keys
    model_repo: Optional[str] = None        # → model.repo
    model_gguf_file: Optional[str] = None   # → model.gguf_file
    # LoRA overrides — mirror lora.* config keys
    lora_repo: Optional[str] = None         # → lora.repo
    lora_strength: Optional[float] = None   # → lora.strength
    # Generation overrides — mirror generation.* config keys
    steps: Optional[int] = None             # → generation.steps
    cfg_scale: Optional[float] = None       # → generation.cfg_scale
    width: Optional[int] = None             # → generation.width
    height: Optional[int] = None            # → generation.height


@router.get("/stats")
def api_stats() -> dict[str, int]:
    return stats()


@router.get("/jobs")
def api_list_jobs() -> list[dict[str, Any]]:
    _heal_stale_running_jobs()
    return list_jobs()


@router.get("/jobs/{job_id}")
def api_get_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # Heal stale running state for this specific job before returning.
    if job.get("status") == "running":
        pid = job.get("worker_pid")
        if pid is not None and not _is_pid_alive(pid):
            msg = f"Worker process (PID {pid}) no longer exists — marked as failed by server."
            append_log(job_id, f"[server] {msg}")
            job = mark_failed(job_id, msg) or job
    return job


@router.post("/jobs", status_code=201)
def api_enqueue(req: EnqueueRequest) -> dict[str, Any]:
    pipeline_cfg = PipelineConfig.from_json(req.config)
    pipeline_cfg.apply_overrides(
        model_repo=req.model_repo,
        model_gguf_file=req.model_gguf_file,
        lora_repo=req.lora_repo,
        lora_strength=req.lora_strength,
        steps=req.steps,
        cfg_scale=req.cfg_scale,
        width=req.width,
        height=req.height,
    )

    # Prepend trigger word automatically if missing from the prompt.
    effective_prompt = req.prompt
    if pipeline_cfg.trigger_word and pipeline_cfg.trigger_word.lower() not in effective_prompt.lower():
        effective_prompt = f"{pipeline_cfg.trigger_word} {effective_prompt}"

    job = enqueue(
        cfg=pipeline_cfg,
        prompt=effective_prompt,
        negative_prompt=req.negative_prompt,
        output=req.output,
    )
    notify.notify()  # wake the in-process worker immediately (no-op in external-worker mode)
    return job


@router.delete("/jobs/{job_id}")
def api_delete_job(job_id: str) -> dict[str, str]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running job")
    delete_job(job_id)
    return {"deleted": job_id}


@router.post("/jobs/{job_id}/retry")
def api_retry_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("failed", "done"):
        raise HTTPException(status_code=409, detail="Only failed or done jobs can be retried")
    # Create a new pending job with the same pipeline config and prompts.
    # Leave the original job untouched.
    pc_dict = job.get("pipeline_config") or {}
    cfg = PipelineConfig.from_dict(pc_dict) if pc_dict else None
    if cfg is None:
        raise HTTPException(status_code=500, detail="Original job has no pipeline config")

    new_job = enqueue(
        cfg=cfg,
        prompt=job.get("prompt", ""),
        negative_prompt=job.get("negative_prompt", ""),
        output=job.get("output"),
    )
    if new_job is None:
        raise HTTPException(status_code=500, detail="Failed to enqueue retry job")
    return new_job


@router.post("/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a running or pending job.

    Sends SIGTERM to the worker process if a PID is stored, then marks the
    job as failed regardless of whether the signal succeeded (the worker
    may already have exited).
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("running", "pending"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job['status']!r} — only running or pending jobs can be cancelled",
        )

    pid = job.get("worker_pid")
    note = ""
    if pid is not None:
        if pid == os.getpid():
            # In-process worker — signal via cancel flag so only the job is
            # interrupted, not the server.  The _progress callback inside
            # process_job picks this up between denoising steps.
            _worker.request_cancel()
            note = "Cancellation requested via cancel flag (in-process worker)."
            append_log(job_id, f"[server] {note}")
        else:
            # External worker process — use SIGTERM as before.
            try:
                os.kill(pid, signal.SIGTERM)
                note = f"SIGTERM sent to PID {pid}."
                append_log(job_id, f"[server] Sent SIGTERM to worker pid {pid}")
            except ProcessLookupError:
                note = f"Worker PID {pid} was already gone."
                append_log(job_id, f"[server] Worker pid {pid} already gone")
            except PermissionError:
                note = f"No permission to signal PID {pid}."
                append_log(job_id, f"[server] Cannot kill pid {pid}: permission denied")
    else:
        note = "No worker PID recorded (job was pending or PID not yet written)."

    updated = mark_failed(job_id, f"Cancelled by user via web UI. {note}")
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to mark job as cancelled")
    return updated


@router.post("/clear-finished")
def api_clear_finished() -> dict[str, int]:
    removed = clear_finished()
    return {"removed": removed}
