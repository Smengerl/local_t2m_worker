"""
Typed configuration container for a fully resolved pipeline configuration.

This module is intentionally free of CLI / argparse dependencies so it can be
imported by any layer (pipelines, worker, generate.py) without pulling in the
CLI machinery.
"""

from typing import Optional


class PipelineConfig:
    """Stores all static pipeline parameters as plain member variables.

    Only contains settings that describe the pipeline itself (model weights,
    dimensions, LoRA, etc.).  Runtime values that vary per request — prompt,
    negative prompt, effective prompt — are passed as explicit arguments to
    ``generate_image()`` and ``print_config()``.

    Usage::

        from cli import build_config, parse_args
        from pipeline_config import PipelineConfig

        cfg, output_path, prompt, negative_prompt = build_config(parse_args())
        print(cfg.model_id, cfg.num_inference_steps)
    """

    def __init__(
        self,
        *,
        pipeline_type: str,
        model_id: str,
        cache_dir: Optional[str] = None,
        output_dir: str,
        num_inference_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        lora_scale: float,
        sequential_cpu_offload: bool,
        adapter_id: Optional[str] = None,
        lora_id: Optional[str] = None,
        trigger_word: Optional[str] = None,
        description: Optional[str] = None,
        true_cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        weight_name: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        gguf_file: Optional[str] = None,
        base_model_id: Optional[str] = None,
    ) -> None:
        # ── Mandatory ────────────────────────────────────────────────────────
        self.pipeline_type: str = pipeline_type.lower()
        self.model_id: str = model_id
        # If None, let Hugging Face / diffusers use its default cache directory
        # (e.g. ~/.cache/huggingface), which allows sharing models between apps.
        self.cache_dir: Optional[str] = cache_dir
        self.output_dir: str = output_dir
        self.num_inference_steps: int = num_inference_steps
        self.guidance_scale: float = guidance_scale
        self.width: int = width
        self.height: int = height
        self.lora_scale: float = lora_scale
        self.sequential_cpu_offload: bool = sequential_cpu_offload
        # ── Optional ─────────────────────────────────────────────────────────
        self.adapter_id: Optional[str] = adapter_id or None
        self.lora_id: Optional[str] = lora_id or None
        self.trigger_word: Optional[str] = trigger_word or None
        self.description: Optional[str] = description or None
        self.true_cfg_scale: Optional[float] = true_cfg_scale
        self.seed: Optional[int] = seed
        self.weight_name: Optional[str] = weight_name or None
        # FLUX-specific: T5 encoder context length (256 = schnell default, 512 = dev)
        self.max_sequence_length: Optional[int] = max_sequence_length or None
        # GGUF-specific: filename of the .gguf quantized transformer in the HF repo
        self.gguf_file: Optional[str] = gguf_file or None
        # GGUF-specific: base model ID supplying text encoders, VAE, scheduler, etc.
        # Required when gguf_file is set. For FLUX.1-dev GGUF fine-tunes this is
        # "black-forest-labs/FLUX.1-dev".
        self.base_model_id: Optional[str] = base_model_id or None

    # ── Convenience ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(pipeline_type={self.pipeline_type!r}, "
            f"model_id={self.model_id!r}, "
            f"steps={self.num_inference_steps}, "
            f"lora_id={self.lora_id!r})"
        )

    def to_dict(self) -> dict:
        """Serialise all fields to a plain dict (JSON-safe)."""
        return {
            "pipeline_type":         self.pipeline_type,
            "model_id":              self.model_id,
            "cache_dir":             self.cache_dir,
            "output_dir":            self.output_dir,
            "num_inference_steps":   self.num_inference_steps,
            "guidance_scale":        self.guidance_scale,
            "width":                 self.width,
            "height":                self.height,
            "lora_scale":            self.lora_scale,
            "sequential_cpu_offload": self.sequential_cpu_offload,
            "adapter_id":            self.adapter_id,
            "lora_id":               self.lora_id,
            "trigger_word":          self.trigger_word,
            "description":           self.description,
            "true_cfg_scale":        self.true_cfg_scale,
            "seed":                  self.seed,
            "weight_name":           self.weight_name,
            "max_sequence_length":   self.max_sequence_length,
            "gguf_file":             self.gguf_file,
            "base_model_id":         self.base_model_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Reconstruct a PipelineConfig from a plain dict (e.g. from JSONL)."""
        return cls(**d)

    def pipeline_cache_key(self) -> tuple:
        """Return a hashable key that uniquely identifies a pipeline instance.

        Two ``PipelineConfig`` objects that produce the same key can safely
        share a loaded pipeline (no model reload needed between jobs).  Every
        parameter that affects the weights loaded into memory is included.
        """
        return (
            self.pipeline_type,
            self.model_id,
            self.adapter_id,
            self.lora_id,
            self.lora_scale,
            self.num_inference_steps,
            self.guidance_scale,
            self.width,
            self.height,
            self.sequential_cpu_offload,
            self.true_cfg_scale,
            self.weight_name,
            self.gguf_file,
            self.base_model_id,
        )
