"""
GET /api/configs  – list available config presets from configs/*.json.
"""

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cli import DEFAULTS as _CLI_DEFAULTS  # noqa: E402

_DEFAULTS_FIELDS = (
    "model_id", "lora_id", "lora_scale",
    "num_inference_steps", "guidance_scale",
    "width", "height", "gguf_file",
)

_EXTRA_FIELDS = ("trigger_word",)


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
            # Collect functional field defaults to populate input placeholders.
            # Merge: global DEFAULTS (lowest priority) ← config file values (higher priority).
            defaults: dict[str, Any] = {}
            for k in _DEFAULTS_FIELDS:
                if k in _CLI_DEFAULTS and _CLI_DEFAULTS[k] is not None:
                    defaults[k] = _CLI_DEFAULTS[k]
                if k in data and data[k] is not None:
                    defaults[k] = data[k]
            # Extra fields exposed directly (not as placeholders)
            extras: dict[str, Any] = {}
            for k in _EXTRA_FIELDS:
                if data.get(k) is not None:
                    extras[k] = data[k]
        except Exception:
            desc = path.stem
            label = path.stem
            hints = {}
            defaults = {}
            extras = {}
        entries.append({
            "value": f"configs/{path.name}",
            "label": label,
            "description": desc,
            "hints": hints,
            "defaults": defaults,
            "extras": extras,
        })
    return entries
