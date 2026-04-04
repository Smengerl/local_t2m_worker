"""
Text-to-Image generation using Hugging Face Diffusers.

Supports:
  - Stable Diffusion 1.5  (~4 GB RAM, runs on MPS/CUDA/CPU)
  - Stable Diffusion XL   (~10 GB RAM, recommended for SDXL LoRAs)
  - Optional LoRA weights  loaded on top of any base model

Run via run.sh or directly:
    python generate.py --config configs/sd15_default.json --prompt "..."
    python generate.py --help
"""

from typing import Optional, Union

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from PIL import Image

from cli import build_config, parse_args, print_config

PipelineType = Union[StableDiffusionPipeline, StableDiffusionXLPipeline]


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        print("Using device: CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using device: MPS (Apple Silicon)")
        return torch.device("mps")
    print("Using device: CPU (this will be slow)")
    return torch.device("cpu")


def _is_xl_model(model_id: str) -> bool:
    """Heuristic: treat a model as SDXL when its ID contains 'xl'."""
    return "xl" in model_id.lower()


def load_pipeline(
    model_id: str,
    device: torch.device,
    cache_dir: str,
    lora_weights: Optional[str] = None,
    lora_scale: float = 0.9,
    sequential_cpu_offload: bool = False,
) -> PipelineType:
    """Load a Stable Diffusion (XL) pipeline and optionally attach LoRA weights.

    Args:
        model_id: Hugging Face repo ID or local path of the base model.
        device: Target torch device.
        cache_dir: Directory used to cache downloaded model weights.
        lora_weights: Hugging Face repo ID / local path of LoRA weights, or
            None to skip LoRA loading.
        lora_scale: Blending scale applied to the LoRA adapter (0–1).
        sequential_cpu_offload: If True, offload model submodules to CPU between
            steps. Reduces peak memory usage at the cost of speed. Recommended
            for SDXL on machines with ≤16 GB unified memory.

    Returns:
        A ready-to-use diffusers pipeline moved to *device*.
    """
    is_xl = _is_xl_model(model_id)
    pipeline_cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

    print(f"Loading {'SDXL' if is_xl else 'SD'} base model: {model_id} ...")

    # float16 on MPS can produce NaN values → black images; float32 is stable
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    kwargs: dict = dict(
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    # Safety checker only exists on SD 1.x pipelines
    if not is_xl:
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

    pipe = pipeline_cls.from_pretrained(model_id, **kwargs)

    if sequential_cpu_offload:
        # enable_sequential_cpu_offload moves submodules to CPU between ops,
        # cutting peak MPS/CUDA memory by ~50 %. Must be called BEFORE .to().
        print("⚙️  Sequential CPU offload enabled (low-VRAM mode).")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)

    pipe.enable_attention_slicing()
    # Tile the VAE decode pass to avoid large contiguous memory allocations.
    # Critical on MPS (Apple Silicon) where SDXL VAE decode easily OOMs.
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    if lora_weights:
        print(f"Loading LoRA weights: {lora_weights}  (scale={lora_scale}) ...")
        # load_lora_weights accepts a repo ID, a local directory, or a single
        # .safetensors / .bin file.
        pipe.load_lora_weights(lora_weights, cache_dir=cache_dir)
        # fuse_lora bakes the LoRA into the model weights with the given scale,
        # which avoids extra overhead during inference.
        pipe.fuse_lora(lora_scale=lora_scale)
        print("LoRA weights fused successfully.")

    return pipe


def generate_image(
    pipe: PipelineType,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 30,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """Run inference and return a PIL image."""
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    )
    return result.images[0]  # type: ignore[index]


def main() -> None:
    args = parse_args()

    # Merge defaults → config file → CLI flags; resolve output path
    cfg, output_path = build_config(args)
    print_config(cfg, args, output_path)

    device = get_device()
    pipe = load_pipeline(
        model_id=cfg["model_id"],
        device=device,
        cache_dir=cfg["cache_dir"],
        lora_weights=cfg["lora_weights"] or None,
        lora_scale=float(cfg["lora_scale"]),
        sequential_cpu_offload=bool(cfg["sequential_cpu_offload"]),
    )

    image = generate_image(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=int(cfg["num_inference_steps"]),
        guidance_scale=float(cfg["guidance_scale"]),
    )

    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")


if __name__ == "__main__":
    main()
