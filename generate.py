"""
Text-to-Image generation entry point.

Delegates all model loading and inference to the pipeline class selected by
the "pipeline_type" key in the config file.  The pipeline_type is mandatory —
every config file must declare it explicitly.

Available pipeline types:
  sd      Stable Diffusion 1.5 / 2.x / 3  (diffusers)
  sdxl    Stable Diffusion XL              (diffusers)
  sd3     Stable Diffusion 3               (diffusers)
  flux    FLUX.1-schnell / dev             (diffusers)
  zimage  Z-Image-Turbo                    (diffusers ZImagePipeline)
  qwen    Qwen-Image                       (diffusers QwenImagePipeline)

Run via run.sh or directly:
    python generate.py --config configs/sd15_default.json "a sunset"
    python generate.py --help
"""

from typing import Any, Callable, Optional
import logging

from cli import build_config, parse_args, print_config
from pipeline_config import PipelineConfig
from pipelines import create_pipeline


def generate_image(
    cfg: PipelineConfig,
    output_path: str,
    prompt: str,
    negative_prompt: str,
    *,
    pipeline_cache: Optional[dict[Any, Any]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Load (or reuse) the pipeline described by *cfg* and generate one image.

    This is the single authoritative place that goes from a fully resolved
    ``PipelineConfig`` to a saved image file.  Both ``main()`` and the batch
    worker call it — neither needs to know about ``create_pipeline()`` directly.

    Args:
        cfg: Fully resolved pipeline configuration (from ``cli.build_config``).
        output_path: Where to save the generated PNG.
        prompt: The prompt passed to the pipeline (trigger word already prepended
            if required).
        negative_prompt: The negative prompt as supplied by the caller.
        pipeline_cache: Optional dict shared across calls.  When the cache key
            derived from *cfg* is already present the loaded pipeline is reused
            and no model reload happens.  If ``None``, a fresh pipeline is
            created for every call.
        progress_callback: Optional callable ``(step, total_steps)`` forwarded
            to ``pipeline.generate()``.  Used by the batch worker to write live
            progress into the queue.

    Returns:
        *output_path* (for the caller's convenience).
    """
    cache_key = cfg.pipeline_cache_key()

    if pipeline_cache is not None and cache_key in pipeline_cache:
        pipeline = pipeline_cache[cache_key]
    else:
        pipeline = create_pipeline(cfg)
        if pipeline_cache is not None:
            if pipeline_cache:
                pipeline_cache.clear()   # free memory before storing new model
            pipeline_cache[cache_key] = pipeline

    image = pipeline.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        progress_callback=progress_callback,
    )
    image.save(output_path)
    return output_path


def main() -> None:
    # Configure logging for direct CLI use so pipeline log.info() calls appear
    # on stdout.  In the batch worker this is handled by basicConfig + the
    # per-job _JobLogHandler; here we just need a minimal StreamHandler.
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    args = parse_args()
    cfg, output_path, effective_prompt, negative_prompt = build_config(args)
    print_config(cfg, output_path, effective_prompt, negative_prompt, args.prompt)
    generate_image(cfg, output_path, effective_prompt, negative_prompt)
    print(f"✅ Image saved to: {output_path}")


if __name__ == "__main__":
    main()
