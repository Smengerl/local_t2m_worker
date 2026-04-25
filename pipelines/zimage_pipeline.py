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

GGUF support:
  - Set gguf_file + components_repo in config to load a quantised transformer.
  - Transformer weights come from the GGUF (e.g. unsloth/Z-Image-Turbo-GGUF).
  - All other components (VAE, text encoder, scheduler, tokenizer) are loaded
    from components_repo (Tongyi-MAI/Z-Image-Turbo).
  - sequential_cpu_offload is incompatible with GGUF quantised tensors.
  - LoRA + GGUF: uses set_adapters() instead of fuse_lora() (GGUF tensors are
    quantised and cannot be modified in-place by fuse_lora()).
"""

import torch
from PIL import Image
from typing import Any, Callable, Optional

from pipelines.base import BasePipeline
from pipeline_config import PipelineConfig

_AnyZImagePipe = Any  # ZImagePipeline


class ZImageBackend(BasePipeline):
    """Diffusers-based backend for Tongyi-MAI/Z-Image-Turbo (and LoRAs).

    Supports both standard bfloat16 loading and GGUF-quantised transformer
    loading (set gguf_file + components_repo in config).
    """

    # Z-Image-Turbo is CFG-distilled → guidance_scale MUST be 0.0.
    # 9 steps → 8 actual DiT forward passes (off-by-one in the scheduler).
    GENERATION_DEFAULTS = {
        "steps":     9,
        "cfg_scale": 0.0,
        "width":     1024,
        "height":    1024,
    }

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

    def _load(self) -> _AnyZImagePipe:
        device = self._get_device()

        # bfloat16 is the recommended dtype for Z-Image-Turbo;
        # fall back to float32 on CPU (no bfloat16 support in many setups).
        dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32

        if self.gguf_file:
            pipe = self._load_gguf(device, dtype)
        else:
            pipe = self._load_standard(device, dtype)

        # ── memory optimisations ──────────────────────────────────────────
        pipe = self._apply_cpu_offload(pipe, device)

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        # ── LoRA ──────────────────────────────────────────────────────────
        self._apply_lora(pipe)

        return pipe

    def _load_standard(self, device: torch.device, dtype: torch.dtype) -> _AnyZImagePipe:
        """Load a standard (non-GGUF) Z-Image pipeline via from_pretrained."""
        from diffusers import ZImagePipeline  # type: ignore[attr-defined]

        self._log(f"Loading Z-Image model: {self.model_id} ...")
        return ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

    def _load_gguf(self, device: torch.device, dtype: torch.dtype) -> _AnyZImagePipe:
        """Load a GGUF-quantised Z-Image transformer and compose a full pipeline.

        The GGUF file contains only the transformer weights.  All other
        components (VAE, text encoder, scheduler, tokenizer) are loaded
        from components_repo (typically Tongyi-MAI/Z-Image-Turbo).

        This follows the same diffusers GGUF loading pattern as FLUX:
            transformer = ZImageTransformer2DModel.from_single_file(gguf_url, ...)
            pipe = ZImagePipeline.from_pretrained(components_repo, transformer=transformer)
        """
        from diffusers import ZImagePipeline  # type: ignore[attr-defined]
        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

        if not self.base_model_id:
            raise ValueError(
                "GGUF loading requires 'components_repo' to be set in the config. "
                "For Z-Image-Turbo GGUF set: "
                '"components_repo": "Tongyi-MAI/Z-Image-Turbo"'
            )

        transformer = self._load_gguf_transformer(ZImageTransformer2DModel, dtype)
        self._log(f"Loading pipeline components from: {self.base_model_id} ...")
        return ZImagePipeline.from_pretrained(
            self.base_model_id,
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )
