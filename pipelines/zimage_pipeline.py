"""
Z-Image pipeline backend (Tongyi-MAI/Z-Image-Turbo and compatible LoRAs).

Z-Image-Turbo is a distilled single-stream DiT model that uses ZImagePipeline
from diffusers (requires diffusers installed from source or ≥ 0.33.0).

Key differences from standard SD/SDXL:
  - guidance_scale MUST be 0.0 for Turbo (CFG is baked into distillation)
  - Optimal steps: 8–16 (default 9 results in 8 DiT forwards)
  - dtype: bfloat16 (float16 can produce artefacts on some hardware)
  - Resolution: up to ~1 MP (1024×1024 or 1152×896 etc.)
  - LoRA loading works identically to diffusers LoRA API

Required config keys:
  model_id              str    HF repo ID or local path (default: Tongyi-MAI/Z-Image-Turbo)
  pipeline_type         str    "zimage"
  lora_id               str|null
  lora_scale            float
  num_inference_steps   int    (8–16 recommended; 9 means 8 DiT forwards)
  guidance_scale        float  (must be 0.0 for Turbo)
  width / height        int
  cache_dir             str
  sequential_cpu_offload bool
"""

import torch
from PIL import Image
from typing import Callable, Optional

from pipelines.base import BasePipeline


class ZImageBackend(BasePipeline):
    """Diffusers-based backend for Tongyi-MAI/Z-Image-Turbo (and LoRAs)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._pipe = self._load(cfg)

    # ── public ───────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Image.Image:
        total = int(self.cfg["num_inference_steps"])
        kwargs: dict = dict(
            prompt=prompt,
            height=int(self.cfg.get("height", 1024)),
            width=int(self.cfg.get("width", 1024)),
            num_inference_steps=total,
            guidance_scale=float(self.cfg["guidance_scale"]),
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        if progress_callback is not None:
            def _cb(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
                progress_callback(step_index + 1, total)
                return callback_kwargs
            kwargs["callback_on_step_end"] = _cb

        result = self._pipe(**kwargs)
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

    def _load(self, cfg: dict):  # type: ignore[return]
        from diffusers import ZImagePipeline  # requires diffusers ≥ 0.33 or source

        model_id: str = cfg.get("model_id") or "Tongyi-MAI/Z-Image-Turbo"
        cache_dir: str = cfg["cache_dir"]
        lora_id: str | None = cfg.get("lora_id") or None
        lora_scale: float = float(cfg.get("lora_scale", 0.9))
        sequential_offload: bool = bool(cfg.get("sequential_cpu_offload", False))

        device = self._get_device()

        # bfloat16 is the recommended dtype for Z-Image-Turbo;
        # fall back to float32 on CPU (no bfloat16 support in many setups)
        dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32

        print(f"Loading Z-Image model: {model_id} ...")
        pipe = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )

        if sequential_offload:
            print("⚙️  Sequential CPU offload enabled (low-VRAM mode).")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        if lora_id:
            print(f"Loading LoRA weights: {lora_id}  (scale={lora_scale}) ...")
            pipe.load_lora_weights(lora_id, cache_dir=cache_dir)
            pipe.fuse_lora(lora_scale=lora_scale)
            print("LoRA weights fused successfully.")

        return pipe
