"""
Text-to-Image generation entry point.

Delegates all model loading and inference to the pipeline class selected by
the "pipeline_type" key in the config file.  The pipeline_type is mandatory —
every config file must declare it explicitly.

Available pipeline types:
  sd          Stable Diffusion 1.5 / 2.x / 3  (diffusers)
  sdxl        Stable Diffusion XL              (diffusers)
  sd3         Stable Diffusion 3               (diffusers)
  flux        FLUX.1-schnell / dev             (diffusers)
  flux2_klein FLUX.2 [klein] 4B distilled      (diffusers)
  lumina2     Lumina-Image-2                   (diffusers)
  zimage      Z-Image-Turbo                    (diffusers ZImagePipeline)
  qwen        Qwen-Image                       (diffusers QwenImagePipeline)

Run via run.sh or directly:
    python generate.py --config configs/sd15_default.json "a sunset"
    python generate.py --help

Offline mode
------------
Pass --offline to run.sh or run_batch_server.sh to set HF_HUB_OFFLINE=1 and
skip all HuggingFace network calls (no HEAD / xet-read-token requests on every
from_pretrained() call).  The model must already be fully cached locally.

    ./run.sh --offline "a sunset"
    ./run_batch_server.sh --offline

Without --offline, diffusers performs its normal cache-validation round-trip
on each load (one HEAD request — no re-download if the model is up to date).
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
        # HF_HUB_OFFLINE is honoured if set externally (e.g. via --offline in
        # run.sh / run_batch_server.sh).  huggingface_hub reads the variable
        # once at import time, so toggling it inside the process is unreliable.
        # Without the flag diffusers performs its normal cache-validation HEAD
        # request on every load — one small network round-trip, no re-download.
        pipeline = create_pipeline(cfg)

        if pipeline_cache is not None:
            if pipeline_cache:
                pipeline_cache.clear()   # free memory before storing new model
                # On MPS (Apple Silicon) the Metal heap is not released by
                # Python GC alone — explicitly empty the MPS allocator cache
                # so the old model's memory is reclaimed *before* the new model
                # is stored.  Without this, both models coexist briefly in the
                # 16 GB unified memory pool and the OS kills the process.
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
            pipeline_cache[cache_key] = pipeline


    if progress_callback is not None:
        progress_callback(0, cfg.num_inference_steps or 0)
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
    import traceback
    args = parse_args()
    cfg, output_path, effective_prompt, negative_prompt = build_config(args)
    print_config(cfg, output_path, effective_prompt, negative_prompt, args.prompt)
    try:
        generate_image(cfg, output_path, effective_prompt, negative_prompt)
        print(f"✅ Image saved to: {output_path}")
    except Exception as e:
        print("❌ Generation failed with exception:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
