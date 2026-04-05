"""
Pipeline factory — maps pipeline_type strings to concrete BasePipeline classes.

Usage:
    from pipelines import create_pipeline

    pipeline = create_pipeline(cfg)
    image = pipeline.generate(prompt, negative_prompt)

Adding a new backend:
  1. Create pipelines/my_pipeline.py with a class inheriting BasePipeline
  2. Add an entry to _REGISTRY below
"""

from pipelines.base import BasePipeline

# Maps the "pipeline_type" config value to its implementation class.
# Import the classes lazily inside create_pipeline to avoid loading heavy
# dependencies (torch, diffusers) until they are actually needed.
_REGISTRY: dict[str, str] = {
    "sd":     "pipelines.sd_pipeline.StableDiffusionBackend",
    "sdxl":   "pipelines.sd_pipeline.StableDiffusionBackend",
    "anima":  "pipelines.anima_pipeline.AnimaPipeline",
    "zimage": "pipelines.zimage_pipeline.ZImageBackend",
}


def create_pipeline(cfg: dict) -> BasePipeline:
    """Instantiate and return the pipeline requested by cfg["pipeline_type"].

    Args:
        cfg: Fully resolved configuration dict (from cli.build_config).
             Must contain the key "pipeline_type".

    Returns:
        A ready-to-use BasePipeline instance.

    Raises:
        KeyError: If "pipeline_type" is missing from cfg.
        ValueError: If the pipeline_type value is not registered.
    """
    pipeline_type: str = cfg["pipeline_type"].lower()

    dotted = _REGISTRY.get(pipeline_type)
    if dotted is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown pipeline_type {pipeline_type!r}. "
            f"Known types: {known}"
        )

    # Lazy import: split "module.path.ClassName" and import on demand
    module_path, class_name = dotted.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls: type[BasePipeline] = getattr(module, class_name)

    return cls(cfg)
