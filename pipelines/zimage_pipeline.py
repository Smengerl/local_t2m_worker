"""
Z-Image pipeline backend (Tongyi-MAI/Z-Image-Turbo and compatibl        self._log(f"Loading Z-Image model: {self.model_id} ...")
        pipe = ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

        if self.sequential_cpu_offload:
            self._log("⚙️  Sequential CPU offload enabled (low-VRAM mode).")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        if self.lora_id:
            self._log(f"Loading LoRA weights: {self.lora_id}  (scale={self.lora_scale}) ...")
            try:
                pipe.load_lora_weights(self.lora_id, cache_dir=self.cache_dir)
                pipe.fuse_lora(lora_scale=self.lora_scale)
                self._log("LoRA weights fused successfully.")-Turbo is a distilled single-stream DiT model that uses ZImagePipeline
from diffusers (requires diffusers installed from source or ≥ 0.33.0).

Key differences from standard SD/SDXL:
  - guidance_scale MUST be 0.0 for Turbo (CFG is baked into distillation)
  - Optimal steps: 8–16 (default 9 results in 8 DiT forwards)
  - dtype: bfloat16 (float16 can produce artefacts on some hardware)
  - Resolution: up to ~1 MP (1024×1024 or 1152×896 etc.)
  - LoRA loading works identically to diffusers LoRA API
"""

import torch
from PIL import Image
from typing import Callable, Optional

from pipelines.base import BasePipeline
from pipeline_config import PipelineConfig


class ZImageBackend(BasePipeline):
    """Diffusers-based backend for Tongyi-MAI/Z-Image-Turbo (and LoRAs)."""

    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg)
        self._pipe = self._load()

    # ── public ───────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Image.Image:
        total = self.num_inference_steps
        kwargs: dict = dict(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=total,
            guidance_scale=self.guidance_scale,
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

    def _load(self):  # type: ignore[return]
        from diffusers import ZImagePipeline  # requires diffusers ≥ 0.33 or source

        device = self._get_device()

        # bfloat16 is the recommended dtype for Z-Image-Turbo;
        # fall back to float32 on CPU (no bfloat16 support in many setups)
        dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32

        self._log(f"Loading Z-Image model: {self.model_id} ...")
        pipe = ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

        if self.sequential_cpu_offload:
            self._log("⚙️  Sequential CPU offload enabled (low-VRAM mode).")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        if self.lora_id:
            self._log(f"Loading LoRA weights: {self.lora_id}  (scale={self.lora_scale}) ...")
            lora_kwargs: dict = {"cache_dir": self.cache_dir}
            if self.weight_name:
                lora_kwargs["weight_name"] = self.weight_name
            try:
                pipe.load_lora_weights(self.lora_id, **lora_kwargs)
                pipe.fuse_lora(lora_scale=self.lora_scale)
                self._log("LoRA weights fused successfully.")
            except ValueError as exc:
                # LoRA target modules don't match this model's architecture.
                # Common cause: LoRA was trained for Flux/Qwen but loaded onto Z-Image-Turbo.
                raise ValueError(
                    f"LoRA '{self.lora_id}' is incompatible with base model '{self.model_id}'.\n"
                    f"The LoRA's target modules do not exist in the base model's architecture.\n"
                    f"This usually means the LoRA was trained for a different model family "
                    f"(e.g. Flux or Qwen) and cannot be used with Z-Image-Turbo.\n"
                    f"Original error: {exc}"
                ) from exc

        return pipe
