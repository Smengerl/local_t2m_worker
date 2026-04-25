"""
GET /api/configs  – list available config presets from configs/*.json.

Response shape per config entry:
  {
    value:       "configs/foo.json",
    label:       "[FLUX] Fast 4-step …",
    description: "Fast 4-step …",
    hints:       {model: "…", lora: "…", generation: "…", system: "…"},
    defaults:    {model_id: "…", lora_id: "…", lora_scale: 0.9, …},
    extras:      {trigger_word: "…"},
    notes:       {about: "…", prompt_guide: "…", warnings: "…"},
  }
"""

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()

_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config_types import ConfigFile

# ── per-request result cache keyed by (sorted file paths × mtimes) ───────────
# Invalidated automatically whenever any .json file under configs/ is added,
# removed, or modified — no server restart required.
_cache_key: tuple | None = None
_cache_result: list[dict[str, Any]] | None = None


def _configs_cache_key() -> tuple:
    """Return a tuple of (path, mtime) pairs for all configs — used as cache key."""
    return tuple(
        (str(p), p.stat().st_mtime)
        for p in sorted(_CONFIGS_DIR.rglob("*.json"))
    )


# ── router ─────────────────────────────────────────────────────────────────────

@router.get("/configs")
def api_list_configs() -> list[dict[str, Any]]:
    """Return all JSON config files as a list of metadata dicts, sorted by filename.

    Configs in subdirectories (e.g. ``configs/nfsw/``) are included and their
    subfolder appears in the dropdown label as a ``[GROUP / …]`` prefix.

    Results are cached in memory and only recomputed when the set of config
    files or their modification times change — no server restart required.
    """
    global _cache_key, _cache_result

    current_key = _configs_cache_key()
    if _cache_result is not None and current_key == _cache_key:
        return _cache_result

    entries: list[dict[str, Any]] = []
    for path in sorted(_CONFIGS_DIR.rglob("*.json")):
        try:
            cfg = ConfigFile.from_json(path, keep_hints=True)

            desc = cfg.description or path.stem
            pipeline = cfg.backend.upper()

            # Subfolder prefix for configs not directly in configs/
            rel = path.relative_to(_CONFIGS_DIR)
            if len(rel.parts) > 1:
                folder = rel.parts[0].upper()
                label = f"[{pipeline} / {folder}] {desc}" if pipeline else f"[{folder}] {desc}"
            else:
                label = f"[{pipeline}] {desc}" if pipeline else desc

            hints = cfg.hints
            defaults: dict[str, Any] = {k: v for k, v in {
                "model_repo":      cfg.model.repo,
                "model_gguf_file": cfg.model.gguf_file,
                "lora_repo":       cfg.lora.repo      if cfg.lora else None,
                "lora_strength":   cfg.lora.strength  if cfg.lora else None,
                "steps":           cfg.generation.steps,
                "cfg_scale":       cfg.generation.cfg_scale,
                "width":           cfg.generation.width,
                "height":          cfg.generation.height,
            }.items() if v is not None}
            extras: dict[str, Any] = (
                {"trigger_word": cfg.lora.trigger}
                if cfg.lora and cfg.lora.trigger
                else {}
            )
            notes = cfg.notes.to_dict() if cfg.notes else None

        except Exception:
            hints, defaults, extras, notes = {}, {}, {}, None
            desc = path.stem
            label = path.stem

        entry: dict[str, Any] = {
            "value":       str(path.relative_to(_CONFIGS_DIR.parent)),
            "label":       label,
            "description": desc,
            "hints":       hints,
            "defaults":    defaults,
            "extras":      extras,
        }
        if notes:
            entry["notes"] = notes
        entries.append(entry)

    _cache_key = current_key
    _cache_result = entries
    return entries
