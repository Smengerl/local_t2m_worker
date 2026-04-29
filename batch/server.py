"""
FastAPI web server for the image-generation batch queue.

Mounts:
  /api/jobs/*      → batch.api.jobs      (job CRUD + cancel/retry)
  /api/configs     → batch.api.configs   (config preset discovery)
  /outputs/*       → batch.api.outputs   (serve generated images)
  /                → batch/static/index.html (browser dashboard)

Usage:
    python -m batch.server              # default: http://127.0.0.1:8000
    python -m batch.server --port 9000
    python -m batch.server --reload     # auto-reload for development
"""

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from batch import notify
from batch import worker as _worker
from batch.api import configs as configs_router
from batch.api import jobs as jobs_router
from batch.api import outputs as outputs_router
from batch.queue import stats as queue_stats, reorder_pending


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Start the in-process worker task; tear it down on server shutdown."""
    task = asyncio.create_task(
        _worker.run_worker_async(keep_alive=True),
        name="image-worker",
    )
    app.state.worker_task = task
    try:
        yield
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_worker._SHUTDOWN_TIMEOUT_S)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        app.state.worker_task = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Image Generation Queue", version="1.0", docs_url="/docs", redoc_url=None, lifespan=lifespan)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(jobs_router.router,    prefix="/api")
app.include_router(configs_router.router, prefix="/api")
app.include_router(outputs_router.router)

# ── Reordering API ───────────────────────────────────────────────────────────
class ReorderRequest(BaseModel):
    job_ids: list[str]

@app.post("/api/jobs/reorder")
async def api_reorder_jobs(req: ReorderRequest):
    """Update the sequence of pending jobs in the queue."""
    reorder_pending(req.job_ids)
    notify.notify()
    return {"status": "ok"}

# ── Frontend ──────────────────────────────────────────────────────────────────
_STATIC = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(_STATIC / "index.html", media_type="text/html")


@app.get("/api/health")
async def api_health(request: Request) -> dict[str, Any]:
    """Return liveness and diagnostic status of server and worker task.

    Fields:
      status          "ok" | "degraded" — degraded means worker is dead.
      worker_alive    True while the background worker task is running.
      worker_error    Exception message if the worker task crashed, else null.
      current_job_id  ID of the job currently being processed, or null if idle.
      pipeline_cached True if a model pipeline is loaded in memory.
      loaded_model    Model repo string of the cached pipeline, or null.
      queue           Job counts per status (pending/running/done/failed).
    """
    task: asyncio.Task | None = getattr(request.app.state, "worker_task", None)
    worker_alive = task is not None and not task.done()

    worker_error: str | None = None
    if task is not None and task.done() and not task.cancelled():
        exc = task.exception()
        if exc is not None:
            worker_error = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok" if worker_alive else "degraded",
        "worker_alive": worker_alive,
        "worker_error": worker_error,
        "current_job_id": _worker._current_job_id,
        "pipeline_cached": _worker._cached_model is not None,
        "loaded_model": _worker._cached_model,
        "queue": queue_stats(),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Start the queue web server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only).")
    args = parser.parse_args()

    # Acquire the exclusive instance lock before starting uvicorn.
    # This prevents running both batch.server and batch.worker simultaneously,
    # which would cause double-generation races on the shared queue.
    from batch.instance_lock import acquire_exclusive
    _instance_lock = acquire_exclusive("server")  # exits if worker/server already running  # noqa: F841

    uvicorn.run("batch.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
