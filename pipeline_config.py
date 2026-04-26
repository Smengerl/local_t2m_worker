"""
Pipeline-ready configuration: extends ConfigFile with a flat property alias
layer and JSONL serialization.

PipelineConfig is a ConfigFile subclass that provides:
  - Flat property aliases for all nested section fields used by pipeline backends
    (e.g. ``cfg.model_id``, ``cfg.num_inference_steps``)
  - JSONL serialization (to_dict / from_dict) for the batch queue

JSONL serialization uses the same flat field names as before to remain
backwards-compatible with existing queue entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from config_types import (
    ConfigFile,
    GenerationConfig,
    LoraConfig,
    ModelConfig,
    NotesConfig,
    SystemConfig,
    _DEFAULT_LORA_STRENGTH,
)


@dataclass
class PipelineConfig(ConfigFile):
    """Fully resolved, pipeline-ready configuration.

    Extends ConfigFile with flat property aliases for all nested section fields
    used throughout the pipeline codebase, and adds JSONL serialization
    (to_dict / from_dict) used by the batch queue.

    Usage::

        from pipeline_config import PipelineConfig

        cfg = PipelineConfig.from_json("configs/flux_schnell.json")
        cfg.apply_overrides(steps=20)
        print(cfg.model_id, cfg.num_inference_steps)
    """

    # ── No extra dataclass fields — all data lives in ConfigFile sections ─────

    # ── Flat property aliases (pipeline codebase uses these names) ────────────

    @property
    def pipeline_type(self) -> str:
        """Alias for ``backend``."""
        return self.backend

    @property
    def model_id(self) -> str:
        """Alias for ``model.repo``."""
        return self.model.repo

    @property
    def cache_dir(self) -> Optional[str]:
        """Alias for ``system.cache_dir``."""
        return self.system.cache_dir

    @property
    def output_dir(self) -> str:
        """Alias for ``system.output_dir``."""
        return self.system.output_dir

    @property
    def num_inference_steps(self) -> Optional[int]:
        """Alias for ``generation.steps`` (None → backend applies its default)."""
        return self.generation.steps

    @property
    def guidance_scale(self) -> Optional[float]:
        """Alias for ``generation.cfg_scale`` (None → backend applies its default)."""
        return self.generation.cfg_scale

    @property
    def width(self) -> Optional[int]:
        """Alias for ``generation.width`` (None → backend applies its default)."""
        return self.generation.width

    @property
    def height(self) -> Optional[int]:
        """Alias for ``generation.height`` (None → backend applies its default)."""
        return self.generation.height

    @property
    def lora_id(self) -> Optional[str]:
        """Alias for ``lora.repo`` (None when no LoRA is configured)."""
        return self.lora.repo if self.lora else None

    @property
    def lora_scale(self) -> float:
        """Alias for ``lora.strength`` (falls back to default when no LoRA)."""
        return self.lora.strength if self.lora else _DEFAULT_LORA_STRENGTH

    @property
    def sequential_cpu_offload(self) -> bool:
        """Alias for ``system.cpu_offload``."""
        return self.system.cpu_offload

    @property
    def trigger_word(self) -> Optional[str]:
        """Alias for ``lora.trigger`` (None when no LoRA is configured)."""
        return self.lora.trigger if self.lora else None

    @property
    def true_cfg_scale(self) -> Optional[float]:
        """Alias for ``generation.cfg_scale_secondary``."""
        return self.generation.cfg_scale_secondary

    @property
    def seed(self) -> Optional[int]:
        """Alias for ``generation.seed``."""
        return self.generation.seed

    @property
    def max_sequence_length(self) -> Optional[int]:
        """Alias for ``generation.max_prompt_tokens``."""
        return self.generation.max_prompt_tokens

    @property
    def gguf_file(self) -> Optional[str]:
        """Alias for ``model.gguf_file``."""
        return self.model.gguf_file

    @property
    def base_model_id(self) -> Optional[str]:
        """Alias for ``model.components_repo``."""
        return self.model.components_repo

    @property
    def weight_name(self) -> Optional[str]:
        """Resolved weight filename: ``lora.file`` takes precedence over ``model.file``."""
        if self.lora and self.lora.file:
            return self.lora.file
        return self.model.file or None

    # ── Convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(backend={self.backend!r}, "
            f"model_id={self.model.repo!r}, "
            f"steps={self.generation.steps}, "
            f"lora_id={self.lora_id!r})"
        )

    # ── JSONL serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict using legacy field names (JSONL-compatible).

        The flat format is kept stable so that existing queue entries written by
        older versions of the code can still be loaded by ``from_dict()``.
        """
        return {
            "pipeline_type":          self.backend,
            "model_id":               self.model.repo,
            "cache_dir":              self.system.cache_dir,
            "output_dir":             self.system.output_dir,
            "num_inference_steps":    self.generation.steps,
            "guidance_scale":         self.generation.cfg_scale,
            "width":                  self.generation.width,
            "height":                 self.generation.height,
            "lora_scale":             self.lora.strength if self.lora else _DEFAULT_LORA_STRENGTH,
            "sequential_cpu_offload": self.system.cpu_offload,
            "lora_id":                self.lora.repo if self.lora else None,
            "trigger_word":           self.lora.trigger if self.lora else None,
            "description":            self.description,
            "true_cfg_scale":         self.generation.cfg_scale_secondary,
            "seed":                   self.generation.seed,
            "weight_name":            self.weight_name,
            "max_sequence_length":    self.generation.max_prompt_tokens,
            "gguf_file":              self.model.gguf_file,
            "base_model_id":          self.model.components_repo,
            "notes":                  self.notes.to_dict() if self.notes else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Reconstruct a PipelineConfig from a flat JSONL dict.

        Accepts the legacy flat format written by ``to_dict()``.  Unknown keys
        are silently ignored so that JSONL jobs written by an older version of
        the code can still be loaded after new fields are added.
        """
        model = ModelConfig(
            repo=d.get("model_id", ""),
            gguf_file=d.get("gguf_file") or None,
            components_repo=d.get("base_model_id") or None,
        )
        lora: Optional[LoraConfig] = None
        if d.get("lora_id"):
            lora = LoraConfig(
                repo=d["lora_id"],
                strength=float(d.get("lora_scale", _DEFAULT_LORA_STRENGTH)),
                trigger=d.get("trigger_word") or None,
            )
        generation = GenerationConfig(
            steps=int(d["num_inference_steps"]) if d.get("num_inference_steps") is not None else None,
            cfg_scale=float(d["guidance_scale"]) if d.get("guidance_scale") is not None else None,
            width=int(d["width"]) if d.get("width") is not None else None,
            height=int(d["height"]) if d.get("height") is not None else None,
            seed=d.get("seed"),
            max_prompt_tokens=d.get("max_sequence_length"),
            cfg_scale_secondary=d.get("true_cfg_scale"),
        )
        system = SystemConfig(
            cpu_offload=bool(d.get("sequential_cpu_offload", False)),
            cache_dir=d.get("cache_dir") or None,
            output_dir=str(d.get("output_dir", "outputs")),
        )
        notes_dict = d.get("notes")
        notes: Optional[NotesConfig] = None
        if isinstance(notes_dict, dict):
            notes = NotesConfig(**{
                k: v for k, v in notes_dict.items()
                if k in ("about", "prompt_guide", "warnings")
            })
        return cls(
            backend=str(d.get("pipeline_type", "sd")),
            model=model,
            description=d.get("description") or None,
            lora=lora,
            generation=generation,
            system=system,
            notes=notes,
        )

    # ── Pipeline identity ─────────────────────────────────────────────────────

    def pipeline_cache_key(self) -> tuple:
        """Return a hashable key that uniquely identifies a pipeline instance.

        Two PipelineConfig objects with the same key can safely share a loaded
        pipeline (no model reload needed between jobs).  Only parameters that
        affect which weights are loaded into memory are included.

        Note: ``seed``, ``steps``, ``cfg_scale``, ``width``, ``height`` are
        intentionally excluded — they are sampling parameters passed at
        ``generate()`` time and do not change which weights are loaded.
        ``notes`` is also excluded as it is GUI metadata only.
        """
        return (
            self.backend,
            self.model.repo,
            self.lora.repo if self.lora else None,
            self.lora.strength if self.lora else None,
            self.system.cpu_offload,
            self.generation.cfg_scale_secondary,
            self.weight_name,
            self.model.gguf_file,
            self.model.components_repo,
        )
