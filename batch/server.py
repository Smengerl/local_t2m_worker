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
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from batch.api import configs as configs_router
from batch.api import jobs as jobs_router
from batch.api import outputs as outputs_router

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Image Generation Queue", version="1.0", docs_url="/docs", redoc_url=None)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(jobs_router.router,    prefix="/api")
app.include_router(configs_router.router, prefix="/api")
app.include_router(outputs_router.router)

# ── Frontend ──────────────────────────────────────────────────────────────────
_STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(_STATIC / "index.html", media_type="text/html")


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Start the queue web server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only).")
    args = parser.parse_args()
    uvicorn.run("batch.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
