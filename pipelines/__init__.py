"""
Pipeline factory — maps pipeline_type strings to concrete BasePipeline classes.

Usage:
    from pipelines import create_pipeline

    pipeline = create_pipeline(
        pipeline_type="sd",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir="models",
        ...
    )
    image = pipeline.generate(prompt, negative_prompt)

Adding a new backend:
  1. Create pipelines/my_pipeline.py with a class inheriting BasePipeline
  2. Add an entry to _REGISTRY below
"""

import importlib
from typing import Optional

from pipeline_config import PipelineConfig
from pipelines.base import BasePipeline

# Maps the "pipeline_type" config value to its implementation class.
# Import the classes lazily inside create_pipeline to avoid loading heavy
# dependencies (torch, diffusers) until they are actually needed.
_REGISTRY: dict[str, str] = {
    "sd":     "pipelines.sd_pipeline.StableDiffusionBackend",
    "sdxl":   "pipelines.sd_pipeline.StableDiffusionBackend",
    "sd3":    "pipelines.sd_pipeline.StableDiffusionBackend",
    "flux":   "pipelines.flux_pipeline.FluxBackend",
    "zimage": "pipelines.zimage_pipeline.ZImageBackend",
    "qwen":   "pipelines.qwen_pipeline.QwenImageBackend",
}


def create_pipeline(cfg: PipelineConfig) -> BasePipeline:
    """Instantiate and return the pipeline described by *cfg*.

    The concrete class is selected from ``_REGISTRY`` using
    ``cfg.pipeline_type`` and loaded lazily to avoid importing heavy
    dependencies (torch, diffusers) until actually needed.

    Args:
        cfg: Fully resolved pipeline configuration.

    Returns:
        A ready-to-use BasePipeline instance.

    Raises:
        ValueError: If ``cfg.pipeline_type`` is not registered.
    """
    dotted = _REGISTRY.get(cfg.pipeline_type)
    if dotted is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown pipeline_type {cfg.pipeline_type!r}. "
            f"Known types: {known}"
        )

    # Lazy import: split "module.path.ClassName" and import on demand
    module_path, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls: type[BasePipeline] = getattr(module, class_name)

    return cls(cfg)
