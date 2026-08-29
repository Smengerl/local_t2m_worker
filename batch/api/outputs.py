"""
GET    /outputs/{filename}  – serve generated images from the outputs/ directory.
DELETE /outputs/{filename}  – delete a generated image file.

Both endpoints accept either a plain filename ("abc.png") or an absolute
path (e.g. "/Users/simon/project/outputs/abc.png").  After resolving to an
absolute path, the file must reside inside the project outputs/ directory;
files outside it are rejected with 403 to prevent path-traversal attacks.
"""

import sys
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch.paths import OUTPUTS_DIR, resolve_within

router = APIRouter()


def _resolve_output_path(raw: str) -> Path:
    """Resolve *raw* to an absolute Path inside the outputs/ directory.

    *raw* may be a plain filename ("abc.png"), a path relative to outputs/
    ("sub/abc.png"), or an absolute path.  Raises HTTPException 403 if the
    resolved path escapes outputs/, or 404 if the file does not exist.
    """
    try:
        return resolve_within(unquote(raw), OUTPUTS_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden: file is outside outputs directory")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")


@router.get("/outputs/{filename:path}")
def serve_output(filename: str) -> FileResponse:
    """Serve a generated image file."""
    path = _resolve_output_path(filename)
    return FileResponse(path)


@router.delete("/outputs/{filename:path}")
def delete_output(filename: str) -> dict[str, str]:
    """Delete a generated image file from the outputs directory."""
    path = _resolve_output_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"deleted": path.name}
