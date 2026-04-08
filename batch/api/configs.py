"""
GET /api/configs  – list available config presets from configs/*.json.
"""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


@router.get("/configs")
def api_list_configs() -> list[dict[str, Any]]:
    """Return all JSON config files as [{value, label, description, hints}], sorted by filename."""
    entries: list[dict[str, Any]] = []
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            desc = data.get("description") or path.stem
            pipeline = data.get("pipeline_type", "")
            label = f"[{pipeline.upper()}] {desc}" if pipeline else desc
            # Collect all _comment_* keys as hints (strip leading "_comment_" prefix for display)
            hints: dict[str, str] = {}
            for k, v in data.items():
                if k.startswith("_comment_") and isinstance(v, str):
                    hint_key = k[len("_comment_"):]  # e.g. "_comment_trigger" → "trigger"
                    hints[hint_key] = v
                elif k == "_comment" and isinstance(v, str):
                    hints[""] = v  # bare _comment shown as a general note
            # Collect functional field defaults to populate input placeholders
            defaults: dict[str, Any] = {
                k: data[k]
                for k in ("model_id", "lora_id", "lora_scale", "num_inference_steps", "guidance_scale")
                if k in data and data[k] is not None
            }
        except Exception:
            desc = path.stem
            label = path.stem
            hints = {}
            defaults = {}
        entries.append({
            "value": f"configs/{path.name}",
            "label": label,
            "description": desc,
            "hints": hints,
            "defaults": defaults,
        })
    return entries
