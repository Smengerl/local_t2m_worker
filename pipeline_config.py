"""
Pipeline-ready configuration: extends ConfigFile with a flat property alias
layer and JSONL serialization.

PipelineConfig is a ConfigFile subclass that provides:
  - Flat property aliases for all nested section fields used by pipeline backends
    (e.g. ``cfg.model_id``, ``cfg.num_inference_steps``)
  - JSONL serialization (to_dict / from_dict) for the batch queue using the
    nested v2 format (matching the ConfigFile section structure)
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
        """Serialise to a nested dict matching the v2 config file structure.

        The nested format mirrors the ConfigFile / section dataclasses so that
        ``from_dict()`` can reconstruct a PipelineConfig without any remapping.
        ``from_dict()`` also accepts the legacy flat format for backward
        compatibility with JSONL entries written by older versions of the code.
        """
        model_dict: dict[str, Any] = {"repo": self.model.repo}
        if self.model.gguf_file:
            model_dict["gguf_file"] = self.model.gguf_file
        if self.model.components_repo:
            model_dict["components_repo"] = self.model.components_repo
        if self.model.file:
            model_dict["file"] = self.model.file

        gen_dict: dict[str, Any] = {
            "steps":               self.generation.steps,
            "cfg_scale":           self.generation.cfg_scale,
            "width":               self.generation.width,
            "height":              self.generation.height,
            "seed":                self.generation.seed,
            "max_prompt_tokens":   self.generation.max_prompt_tokens,
            "cfg_scale_secondary": self.generation.cfg_scale_secondary,
        }

        system_dict: dict[str, Any] = {
            "cpu_offload": self.system.cpu_offload,
            "cache_dir":   self.system.cache_dir,
            "output_dir":  self.system.output_dir,
        }

        result: dict[str, Any] = {
            "backend":     self.backend,
            "description": self.description,
            "model":       model_dict,
            "generation":  gen_dict,
            "system":      system_dict,
        }

        if self.lora:
            lora_dict: dict[str, Any] = {
                "repo":     self.lora.repo,
                "strength": self.lora.strength,
            }
            if self.lora.file:
                lora_dict["file"] = self.lora.file
            if self.lora.trigger:
                lora_dict["trigger"] = self.lora.trigger
            result["lora"] = lora_dict

        if self.notes:
            result["notes"] = self.notes.to_dict()

        return result

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Reconstruct a PipelineConfig from a nested v2 dict (as written by ``to_dict()``).

        Args:
            d: Dict produced by ``to_dict()``.  Must use the nested v2 format
               with ``model``, ``generation``, and ``system`` sub-dicts.

        Returns:
            Reconstructed PipelineConfig.
        """
        m = d["model"]
        model = ModelConfig(
            repo=m.get("repo", ""),
            gguf_file=m.get("gguf_file") or None,
            components_repo=m.get("components_repo") or None,
            file=m.get("file") or None,
        )
        lora: Optional[LoraConfig] = None
        lo = d.get("lora")
        if lo and isinstance(lo, dict):
            lora = LoraConfig(
                repo=lo["repo"],
                strength=float(lo.get("strength", _DEFAULT_LORA_STRENGTH)),
                file=lo.get("file") or None,
                trigger=lo.get("trigger") or None,
            )
        g = d.get("generation") or {}
        generation = GenerationConfig(
            steps=int(g["steps"]) if g.get("steps") is not None else None,
            cfg_scale=float(g["cfg_scale"]) if g.get("cfg_scale") is not None else None,
            width=int(g["width"]) if g.get("width") is not None else None,
            height=int(g["height"]) if g.get("height") is not None else None,
            seed=g.get("seed"),
            max_prompt_tokens=g.get("max_prompt_tokens"),
            cfg_scale_secondary=g.get("cfg_scale_secondary"),
        )
        s = d.get("system") or {}
        system = SystemConfig(
            cpu_offload=bool(s.get("cpu_offload", False)),
            cache_dir=s.get("cache_dir") or None,
            output_dir=str(s.get("output_dir", "outputs")),
        )
        notes_dict = d.get("notes")
        notes: Optional[NotesConfig] = None
        if isinstance(notes_dict, dict):
            notes = NotesConfig(**{
                k: v for k, v in notes_dict.items()
                if k in ("about", "prompt_guide", "warnings")
            })
        return cls(
            backend=str(d.get("backend", "sd")),
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
