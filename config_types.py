"""Typed dataclasses for the v2 config file format.

These represent the fully parsed config file sections *before* CLI overrides
are applied.  The final pipeline-ready representation is ``PipelineConfig``
(see ``pipeline_config.py``), which extends this class with flat property
aliases for all nested section fields.

Typical usage::

    from config_types import ConfigFile

    cfg = ConfigFile.from_json("configs/flux_schnell.json")
    cfg.apply_overrides(steps=8, cfg_scale=0.0)   # apply CLI overrides
    pipeline_cfg = PipelineConfig.from_config_file(cfg)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Self

# ── Section-level defaults that are referenced in more than one place ─────────
_DEFAULT_LORA_STRENGTH: float = 0.9


# ── Section dataclasses ───────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Config file ``model`` section."""

    repo: str
    gguf_file: Optional[str] = None
    components_repo: Optional[str] = None  # → PipelineConfig.base_model_id
    file: Optional[str] = None             # → PipelineConfig.weight_name (non-LoRA)


@dataclass
class LoraConfig:
    """Config file ``lora`` section."""

    repo: str
    file: Optional[str] = None              # → PipelineConfig.weight_name (takes precedence over model.file)
    strength: float = _DEFAULT_LORA_STRENGTH  # → PipelineConfig.lora_scale
    trigger: Optional[str] = None           # → PipelineConfig.trigger_word


@dataclass
class GenerationConfig:
    """Config file ``generation`` section.

    All numeric fields are ``Optional`` — when ``None``, the backend class
    applies its own ``GENERATION_DEFAULTS`` at pipeline initialisation time.
    This keeps config files minimal: only values that intentionally differ
    from the backend's defaults need to be specified.
    """

    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    max_prompt_tokens: Optional[int] = None     # → PipelineConfig.max_sequence_length
    cfg_scale_secondary: Optional[float] = None # → PipelineConfig.true_cfg_scale


@dataclass
class SystemConfig:
    """Config file ``system`` section."""

    cpu_offload: bool = False
    cache_dir: Optional[str] = None
    output_dir: str = "outputs"


@dataclass
class NotesConfig:
    """Config file ``notes`` section — GUI / dashboard metadata only, not used by the pipeline."""

    about: Optional[str] = None
    prompt_guide: Optional[str] = None
    warnings: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        """Return non-None fields as a plain dict (for PipelineConfig.notes)."""
        return {k: v for k, v in vars(self).items() if v is not None}


# ── Top-level config file object ──────────────────────────────────────────────

@dataclass
class ConfigFile:
    """Fully parsed v2 config JSON file as typed Python objects.

    This is the internal data representation produced by ``from_json()`` (or
    ``default()`` when no config file is given) that travels through
    CLI-override application before being converted to a ``PipelineConfig``
    via ``PipelineConfig.from_config_file()``.

    Attributes:
        backend:     Pipeline backend identifier (e.g. "flux", "sd", "sdxl").
                     Maps to ``PipelineConfig.pipeline_type``.
        model:       Model source and file configuration.
        description: Human-readable label shown in the dashboard dropdown.
        lora:        LoRA adapter configuration, or None when not used.
        generation:  Sampling parameters (steps, guidance scale, dimensions, …).
        system:      Runtime settings (CPU offload, cache/output directories).
        notes:       GUI display metadata (about, prompt guide, warnings).
        hints:       Per-section ``_hint`` tooltip strings from the config file,
                     keyed by section name (``model``, ``lora``, ``generation``,
                     ``system``).  Only populated when ``from_json()`` is called
                     with ``keep_hints=True`` (GUI context only — never set
                     during CLI generation or batch enqueueing).
    """

    backend: str
    model: ModelConfig
    description: Optional[str] = None
    lora: Optional[LoraConfig] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    notes: Optional[NotesConfig] = None
    hints: dict[str, str] = field(default_factory=dict)

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "ConfigFile":
        """Return a ConfigFile populated with built-in defaults (no JSON file needed).

        Used by ``build_config()`` when the user runs without ``--config``.
        """
        return cls(
            backend="sd",
            model=ModelConfig(repo="stable-diffusion-v1-5/stable-diffusion-v1-5"),
        )

    @classmethod
    def from_json(cls, path: str | Path, *, keep_hints: bool = False) -> Self:
        """Parse a v2 JSON config file into a typed ConfigFile.

        Annotation-only keys (``_hint``, ``_comment*``) are stripped at every
        level before parsing.  Field defaults from the dataclasses are applied
        for any optional keys absent in the file.

        Args:
            path:       Path to the JSON config file.
            keep_hints: When ``True``, ``_hint`` strings from each section are
                        collected into ``ConfigFile.hints`` instead of being
                        discarded.  Use this in GUI contexts (e.g. the batch
                        server's ``/api/configs`` endpoint) to populate tooltip
                        text.  Must be ``False`` (the default) for CLI
                        generation and batch enqueueing so that hint data never
                        reaches the pipeline or the JSONL queue.

        Returns:
            Parsed ConfigFile with all optional fields defaulted.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the required fields ``backend`` or ``model.repo``
                are missing or have the wrong type.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw: dict[str, Any] = json.loads(p.read_text())

        # ── Collect hints before stripping (GUI context only) ─────────────────
        # _hint keys are annotation-only and stripped below; read them first
        # so they can optionally be stored in ConfigFile.hints.
        hints: dict[str, str] = {}
        if keep_hints:
            for section in ("model", "lora", "generation", "system"):
                sec = raw.get(section)
                if isinstance(sec, dict):
                    h = sec.get("_hint")
                    if h and isinstance(h, str):
                        hints[section] = h

        def _strip(d: dict[str, Any]) -> dict[str, Any]:
            """Remove annotation-only keys (_hint, _comment*) from a dict."""
            return {k: v for k, v in d.items() if not k.startswith("_")}

        raw = _strip(raw)

        # ── Validate required top-level fields ────────────────────────────────
        if not raw.get("backend"):
            raise ValueError(
                f"{path}: missing required key 'backend' "
                "(e.g. \"backend\": \"flux\"). "
                "All configs must use the v2 nested format."
            )
        raw_model = raw.get("model")
        if not isinstance(raw_model, dict) or not raw_model.get("repo"):
            raise ValueError(
                f"{path}: missing required key 'model.repo' "
                "(e.g. \"model\": {{\"repo\": \"org/model-name\"}}). "
                "All configs must use the v2 nested format."
            )

        # ── model ─────────────────────────────────────────────────────────────
        m = _strip(raw_model)
        model = ModelConfig(
            repo=m["repo"],
            gguf_file=m.get("gguf_file") or None,
            components_repo=m.get("components_repo") or None,
            file=m.get("file") or None,
        )

        # ── lora (optional — omit entirely when not used) ─────────────────────
        lora: Optional[LoraConfig] = None
        raw_lora = raw.get("lora")
        if isinstance(raw_lora, dict):
            lo = _strip(raw_lora)
            lora = LoraConfig(
                repo=lo.get("repo") or "",
                file=lo.get("file") or None,
                strength=float(lo["strength"]) if "strength" in lo else _DEFAULT_LORA_STRENGTH,
                trigger=lo.get("trigger") or None,
            )

        # ── generation ────────────────────────────────────────────────────────
        _gen = GenerationConfig()   # instance carries field defaults
        raw_gen = _strip(raw.get("generation") or {})
        generation = GenerationConfig(
            steps=int(raw_gen["steps"]) if "steps" in raw_gen else _gen.steps,
            cfg_scale=float(raw_gen["cfg_scale"]) if "cfg_scale" in raw_gen else _gen.cfg_scale,
            width=int(raw_gen["width"]) if "width" in raw_gen else _gen.width,
            height=int(raw_gen["height"]) if "height" in raw_gen else _gen.height,
            seed=int(raw_gen["seed"]) if raw_gen.get("seed") is not None else None,
            max_prompt_tokens=int(raw_gen["max_prompt_tokens"]) if "max_prompt_tokens" in raw_gen else None,
            cfg_scale_secondary=float(raw_gen["cfg_scale_secondary"]) if "cfg_scale_secondary" in raw_gen else None,
        )

        # ── system ────────────────────────────────────────────────────────────
        _sys = SystemConfig()       # instance carries field defaults
        raw_sys = _strip(raw.get("system") or {})
        system = SystemConfig(
            cpu_offload=bool(raw_sys["cpu_offload"]) if "cpu_offload" in raw_sys else _sys.cpu_offload,
            cache_dir=raw_sys.get("cache_dir") or None,
            output_dir=str(raw_sys["output_dir"]) if "output_dir" in raw_sys else _sys.output_dir,
        )

        # ── notes (optional GUI metadata) ─────────────────────────────────────
        notes: Optional[NotesConfig] = None
        raw_notes = raw.get("notes")
        if isinstance(raw_notes, dict):
            n = _strip(raw_notes)
            notes = NotesConfig(
                about=n.get("about") or None,
                prompt_guide=n.get("prompt_guide") or None,
                warnings=n.get("warnings") or None,
            )

        return cls(
            backend=str(raw["backend"]).lower(),
            model=model,
            description=raw.get("description") or None,
            lora=lora,
            generation=generation,
            system=system,
            notes=notes,
            hints=hints,
        )

    # ── Override application ──────────────────────────────────────────────────

    def apply_overrides(
        self,
        *,
        model_repo: Optional[str] = None,
        model_gguf_file: Optional[str] = None,
        lora_repo: Optional[str] = None,
        lora_strength: Optional[float] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Patch config sections in-place with any non-None override values.

        Mirrors the CLI flag naming convention: parameter names here match
        the ``--flag-name`` CLI flags (with hyphens replaced by underscores),
        which in turn mirror the ``section.field`` config file key paths.

        A minimal ``LoraConfig`` is created automatically when a LoRA override
        is supplied but the config file had no ``lora`` section.

        Args:
            model_repo:      Override for ``model.repo``.
            model_gguf_file: Override for ``model.gguf_file``.
            lora_repo:       Override for ``lora.repo``.
            lora_strength:   Override for ``lora.strength``.
            steps:           Override for ``generation.steps``.
            cfg_scale:       Override for ``generation.cfg_scale``.
            width:           Override for ``generation.width``.
            height:          Override for ``generation.height``.
            output_dir:      Override for ``system.output_dir``.
            cache_dir:       Override for ``system.cache_dir``.
        """
        if model_repo is not None:
            self.model.repo = model_repo
        if model_gguf_file is not None:
            self.model.gguf_file = model_gguf_file

        if lora_repo is not None:
            if self.lora is None:
                self.lora = LoraConfig(repo=lora_repo)
            else:
                self.lora.repo = lora_repo
        if lora_strength is not None:
            if self.lora is None:
                self.lora = LoraConfig(repo="", strength=lora_strength)
            else:
                self.lora.strength = lora_strength

        if steps is not None:
            self.generation.steps = steps
        if cfg_scale is not None:
            self.generation.cfg_scale = cfg_scale
        if width is not None:
            self.generation.width = width
        if height is not None:
            self.generation.height = height
        if output_dir is not None:
            self.system.output_dir = output_dir
        if cache_dir is not None:
            self.system.cache_dir = cache_dir
