"""
Stable Diffusion pipeline backend (SD 1.5 and SDXL).

Handles both pipeline variants via a model_id heuristic:
  - model_id containing "xl"  →  StableDiffusionXLPipeline
  - everything else           →  StableDiffusionPipeline

Supports optional LoRA weights and memory-saving CPU offload mode.

Required config keys:
  model_id              str   HF repo ID or local path of the base model
  pipeline_type         str   "sd" or "sdxl"
  adapter_id            str|null  HF repo ID / local path for an adapter (ControlNet, refiner, …), or null
  lora_id               str|null  HF repo ID / local path for LoRA weights, or null
  lora_scale            float  LoRA blending strength (0–1)
  num_inference_steps   int
  guidance_scale        float
  width                 int
  height                int
  cache_dir             str
  sequential_cpu_offload bool  enable on MPS/CUDA with ≤16 GB unified memory
"""

from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from pipelines.base import BasePipeline

_DiffusersPipe = StableDiffusionPipeline | StableDiffusionXLPipeline


class StableDiffusionBackend(BasePipeline):
    """Diffusers-based backend for SD 1.5 and Stable Diffusion XL."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._pipe = self._load(cfg)

    # ── public ───────────────────────────────────────────────────────────────

    def generate(self, prompt: str, negative_prompt: str = "") -> Image.Image:
        result = self._pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(self.cfg["num_inference_steps"]),
            guidance_scale=float(self.cfg["guidance_scale"]),
            width=int(self.cfg.get("width", 512)),
            height=int(self.cfg.get("height", 512)),
        )
        return result.images[0]  # type: ignore[index]

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            print("Using device: CUDA")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("Using device: MPS (Apple Silicon)")
            return torch.device("mps")
        print("Using device: CPU (this will be slow)")
        return torch.device("cpu")

    @staticmethod
    def _is_xl(model_id: str) -> bool:
        return "xl" in model_id.lower()

    def _load(self, cfg: dict) -> _DiffusersPipe:
        model_id: str = cfg["model_id"]
        cache_dir: str = cfg["cache_dir"]
        adapter_id: Optional[str] = cfg.get("adapter_id") or None
        lora_id: Optional[str] = cfg.get("lora_id") or None
        lora_scale: float = float(cfg.get("lora_scale", 0.9))
        sequential_offload: bool = bool(cfg.get("sequential_cpu_offload", False))

        is_xl = self._is_xl(model_id)
        pipeline_cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
        label = "SDXL" if is_xl else "SD"
        device = self._get_device()

        print(f"Loading {label} base model: {model_id} ...")

        # float16 on MPS can produce NaN values → black images; float32 is stable
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        kwargs: dict = dict(torch_dtype=dtype, cache_dir=cache_dir)
        if not is_xl:
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

        pipe = pipeline_cls.from_pretrained(model_id, **kwargs)

        if sequential_offload:
            # Offloads submodules to CPU between ops; cuts peak memory ~50 %.
            # Must be called before .to(device).
            print("⚙️  Sequential CPU offload enabled (low-VRAM mode).")
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)

        pipe.enable_attention_slicing()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        if lora_id:
            print(f"Loading LoRA weights: {lora_id}  (scale={lora_scale}) ...")
            pipe.load_lora_weights(lora_id, cache_dir=cache_dir)
            pipe.fuse_lora(lora_scale=lora_scale)
            print("LoRA weights fused successfully.")

        return pipe
