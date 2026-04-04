"""
Text-to-Image generation using Hugging Face Diffusers.
Uses Stable Diffusion 1.5 by default (~4 GB RAM, runs on MPS/CUDA/CPU).
To use FLUX models, you need at least 24 GB RAM.
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────
# Default: SD 1.5 — public, no token needed, ~4 GB RAM
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# FLUX models (require 24+ GB RAM — NOT suitable for 16 GB machines):
# MODEL_ID = "kpsss34/FHDR_Uncensored"

CACHE_DIR = "models"
OUTPUT_DIR = "outputs"

PROMPT = "A futuristic city at sunset, digital art, highly detailed"
NEGATIVE_PROMPT = "blurry, low quality, deformed, ugly"

NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
# ─────────────────────────────────────────────────────────────────────────────


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


def load_pipeline(model_id: str, device: torch.device) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline."""
    print(f"Loading model: {model_id} ...")
    # float16 on MPS can produce NaN values → black images; float32 is stable
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    pipe = load_pipeline(MODEL_ID, device)

    print(f"Prompt: {PROMPT!r}")
    image = generate_image(
        pipe,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )

    output_path = os.path.join(OUTPUT_DIR, "output.png")
    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")


if __name__ == "__main__":
    main()
