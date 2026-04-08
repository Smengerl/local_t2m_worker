"""
Qwen-Image pipeline backend (Qwen/Qwen-Image and compatible LoRAs).

Qwen-Image is a diffusion foundation model from Alibaba/Qwen that uses the
QwenImagePipeline from diffusers (requires diffusers installed from source or
≥ 0.34.0).

Key differences from standard SD/SDXL or Z-Image-Turbo:
  - Uses DiffusionPipeline (resolves to QwenImagePipeline automatically)
  - true_cfg_scale (not guidance_scale) controls CFG strength (default 4.0)
  - guidance_scale is NOT used (ignored if present in config)
  - Optimal steps: 50 (not distilled, no few-step shortcut)
  - dtype: bfloat16 on CUDA/MPS, float32 on CPU
  - Native resolutions: 1328×1328 (1:1), 1664×928 (16:9), 928×1664 (9:16), etc.
  - LoRA loading: standard diffusers load_lora_weights API
"""

import torch
from PIL import Image
from typing import Callable, Optional

from pipelines.base import BasePipeline
from pipeline_config import PipelineConfig


class QwenImageBackend(BasePipeline):
    """Diffusers-based backend for Qwen/Qwen-Image (and compatible LoRAs)."""

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
        # Qwen-Image uses true_cfg_scale; fall back to guidance_scale for
        # configs that still specify only guidance_scale.
        true_cfg = self.true_cfg_scale or self.guidance_scale
        kwargs: dict = dict(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=total,
            true_cfg_scale=true_cfg,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        if progress_callback is not None:
            # Signal total immediately so the web-UI shows the progress bar
            # before the first denoising step completes.
            progress_callback(0, total)
            def _cb(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
                progress_callback(step_index + 1, total)
                return callback_kwargs
            kwargs["callback_on_step_end"] = _cb

        result = self._pipe(**kwargs)
        return result.images[0]  # type: ignore[index]

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self):  # type: ignore[return]
        from diffusers import DiffusionPipeline  # resolves to QwenImagePipeline

        device = self._get_device()

        # bfloat16 is recommended; fall back to float32 on CPU.
        # Qwen-Image is a 20B model (~40 GB in bfloat16 / ~20 GB in float16).
        # On MPS with ≤16 GB unified memory we must use float16 and CPU offload.
        # float16 is preferred over bfloat16 on MPS: MPS has limited bfloat16
        # support and silently upcasts some ops to float32, doubling memory use.
        if self.sequential_cpu_offload and device.type == "mps":
            # Load directly onto CPU in float16 so we never spike past RAM limit.
            # enable_sequential_cpu_offload() will move each layer to MPS just-
            # in-time during the forward pass.
            load_device = torch.device("cpu")
            dtype = torch.float16
            self._log("⚙️  Low-memory mode: loading in float16 on CPU for sequential MPS offload.")
        elif device.type == "mps":
            load_device = device
            dtype = torch.float16  # float16 is safer than bfloat16 on MPS
        elif device.type == "cuda":
            load_device = device
            dtype = torch.bfloat16
        else:
            load_device = device
            dtype = torch.float32

        self._log(f"Loading Qwen-Image model: {self.model_id}  dtype={dtype} ...")
        pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

        if self.sequential_cpu_offload:
            if device.type == "mps":
                # enable_sequential_cpu_offload moves each submodule to MPS
                # one at a time during inference — minimal peak MPS usage.
                self._log("⚙️  Sequential CPU offload enabled (MPS low-memory mode).")
                pipe.enable_sequential_cpu_offload()
            else:
                # On CUDA, enable_model_cpu_offload is the standard approach.
                self._log("⚙️  Model CPU offload enabled (CUDA low-VRAM mode).")
                pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(load_device)

        if self.lora_id:
            self._log(f"Loading LoRA weights: {self.lora_id}  (scale={self.lora_scale}) ...")
            try:
                pipe.load_lora_weights(self.lora_id, cache_dir=self.cache_dir)
                pipe.fuse_lora(lora_scale=self.lora_scale)
                self._log("LoRA weights fused successfully.")
            except ValueError as exc:
                raise ValueError(
                    f"LoRA '{self.lora_id}' is incompatible with base model '{self.model_id}'.\n"
                    f"The LoRA's target modules do not exist in the base model's architecture.\n"
                    f"This usually means the LoRA was trained for a different model family.\n"
                    f"Original error: {exc}"
                ) from exc

        return pipe
