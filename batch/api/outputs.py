"""
GET    /outputs/{filename}  – serve generated images from the outputs/ directory.
DELETE /outputs/{filename}  – delete a generated image file.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


@router.get("/outputs/{filename}")
def serve_output(filename: str) -> FileResponse:
    # Prevent path traversal attacks.
    path = (_OUTPUTS_DIR / filename).resolve()
    if not str(path).startswith(str(_OUTPUTS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@router.delete("/outputs/{filename}")
def delete_output(filename: str) -> dict[str, str]:
    """Delete a generated image file from the outputs directory."""
    path = (_OUTPUTS_DIR / filename).resolve()
    if not str(path).startswith(str(_OUTPUTS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"deleted": filename}
