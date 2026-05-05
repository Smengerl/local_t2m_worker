"""
GET    /outputs/{filename}  – serve generated images from the outputs/ directory.
DELETE /outputs/{filename}  – delete a generated image file.

Both endpoints accept either a plain filename ("abc.png") or an absolute
path (e.g. "/Users/simon/project/examples/abc.png").  After resolving to an
absolute path, the file must reside inside the project outputs/ directory;
files outside it are rejected with 403 to prevent path-traversal attacks.
"""

from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_OUTPUTS_DIR = (Path(__file__).parent.parent.parent / "outputs").resolve()
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def _resolve_output_path(raw: str) -> Path:
    """Resolve *raw* to an absolute Path and enforce outputs-dir containment.

    *raw* may be:
    - a plain filename: ``"abc123.png"``  → resolved inside outputs/
    - a project-relative path: ``"outputs/abc123.png"``  → resolved from project root
    - an absolute path: ``"/Users/simon/.../outputs/abc123.png"``  → used as-is

    Raises HTTPException 403 if the resolved path is outside ``_OUTPUTS_DIR``,
    or 404 if the file does not exist.
    """
    decoded = unquote(raw)
    candidate = Path(decoded)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        # Try resolving from project root first (handles "outputs/foo.png"),
        # fall back to resolving directly inside _OUTPUTS_DIR (plain "foo.png").
        from_root = (_PROJECT_ROOT / candidate).resolve()
        resolved = from_root if from_root.exists() else (_OUTPUTS_DIR / candidate).resolve()

    if not str(resolved).startswith(str(_OUTPUTS_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden: file is outside outputs directory")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return resolved


@router.get("/outputs/{filename:path}")
def serve_output(filename: str) -> FileResponse:
    """Serve a generated image file."""
    path = _resolve_output_path(filename)
    return FileResponse(path)


@router.delete("/outputs/{filename:path}")
def delete_output(filename: str) -> dict[str, str]:
    """Delete a generated image file from the outputs directory."""
    path = _resolve_output_path(filename)
    path.unlink()
    return {"deleted": path.name}
