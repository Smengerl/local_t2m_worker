"""
FLUX.1 pipeline backend (black-forest-labs/FLUX.1-schnell and FLUX.1-dev).

Uses diffusers FluxPipeline. Key characteristics:
  - 12-billion parameter rectified-flow transformer
  - guidance_scale MUST be 0.0 for FLUX.1-schnell (CFG-distilled)
  - FLUX.1-dev supports positive guidance_scale (3.5–7.0 recommended)
  - max_sequence_length controls T5 encoder context (default 256)
  - Sequential CPU offload is strongly recommended on ≤16 GB unified memory
"""

from typing import Callable, Optional

import torch
from diffusers import FluxPipeline
from PIL import Image

from pipeline_config import PipelineConfig
from pipelines.base import BasePipeline


class FluxBackend(BasePipeline):
    """Diffusers-based backend for FLUX.1-schnell and FLUX.1-dev."""

    # T5 context length: 256 is the model card default for schnell.
    # dev benefits from up to 512 for complex prompts.
    _MAX_SEQUENCE_LENGTH: int = 256

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

        # FLUX-schnell is CFG-distilled → guidance_scale must be 0.0.
        # Negative prompts have no effect at guidance_scale=0.0 but are
        # silently ignored by FluxPipeline, so we still pass them harmlessly.
        kwargs: dict = dict(
            prompt=prompt,
            num_inference_steps=total,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
            max_sequence_length=self._MAX_SEQUENCE_LENGTH,
        )

        if progress_callback is not None:
            def _cb(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
                progress_callback(step_index + 1, total)
                return callback_kwargs
            kwargs["callback_on_step_end"] = _cb

        result = self._pipe(**kwargs)
        return result.images[0]  # type: ignore[index]

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self) -> FluxPipeline:
        device = self._get_device()

        # bfloat16 is the recommended dtype for FLUX on both CUDA and MPS.
        # It avoids NaN instabilities that can occur with float16 in the
        # FLUX transformer, while using half the memory of float32.
        dtype = torch.bfloat16

        self._log(f"Loading FLUX base model: {self.model_id} ...")

        pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

        if self.sequential_cpu_offload:
            # Strongly recommended for FLUX on 16 GB machines: the transformer
            # alone is ~24 GB in fp32; bfloat16 + offloading brings peak use
            # within 16 GB unified memory.
            self._log("⚙️  Sequential CPU offload enabled (low-VRAM mode).")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        if self.lora_id:
            self._log(f"Loading LoRA weights: {self.lora_id}  (scale={self.lora_scale}) ...")
            lora_kwargs: dict = {"cache_dir": self.cache_dir}
            if self.weight_name:
                lora_kwargs["weight_name"] = self.weight_name
                self._log(f"  Using specific weight file: {self.weight_name}")
            pipe.load_lora_weights(self.lora_id, **lora_kwargs)
            pipe.fuse_lora(lora_scale=self.lora_scale)
            self._log("LoRA weights fused successfully.")

        return pipe
