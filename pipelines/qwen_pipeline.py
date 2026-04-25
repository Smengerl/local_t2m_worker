"""
Qwen-Image pipeline backend (Qwen/Qwen-Image-2512 and compatible LoRAs).

Qwen-Image is a diffusion foundation model from Alibaba/Qwen that uses the
QwenImagePipeline from diffusers (requires diffusers installed from source or
≥ 0.34.0).

Key differences from standard SD/SDXL or Z-Image-Turbo:
  - Uses QwenImagePipeline (or DiffusionPipeline which auto-resolves to it)
  - true_cfg_scale (not guidance_scale) controls CFG strength (default 4.0)
  - guidance_scale is NOT used (ignored if present in config)
  - Optimal steps: 50 (not distilled, no few-step shortcut)
  - dtype: bfloat16 on CUDA/MPS, float32 on CPU
  - Native resolutions: 1328×1328 (1:1), 1664×928 (16:9), 928×1664 (9:16), etc.
  - LoRA loading: standard diffusers load_lora_weights API

GGUF support:
  - Set gguf_file + components_repo in config to load a quantised transformer.
  - Transformer weights come from the GGUF (e.g. unsloth/Qwen-Image-2512-GGUF).
  - All other components (VAE, text encoder, scheduler, tokenizer) are loaded
    from components_repo (Qwen/Qwen-Image-2512).
  - Q4_0 (~11.9 GB) fits comfortably in 16 GB unified memory without offloading.
  - sequential_cpu_offload is incompatible with GGUF quantised tensors.
  - LoRA + GGUF: uses set_adapters() instead of fuse_lora() (GGUF tensors are
    quantised and cannot be modified in-place by fuse_lora()).
"""

import torch
from PIL import Image
from typing import Any, Callable, Optional

from pipelines.base import BasePipeline
from pipeline_config import PipelineConfig

_AnyQwenPipe = Any  # QwenImagePipeline


class QwenImageBackend(BasePipeline):
    """Diffusers-based backend for Qwen/Qwen-Image-2512 (and compatible LoRAs).

    Supports both standard bfloat16/float16 loading and GGUF-quantised
    transformer loading (set gguf_file + components_repo in config).
    """

    # Qwen-Image uses true_cfg_scale (not guidance_scale); 50 steps recommended.
    # Native 1:1 resolution is 1328×1328.
    GENERATION_DEFAULTS = {
        "steps":     50,
        "cfg_scale": 4.0,   # used as true_cfg_scale fallback; guidance_scale is ignored
        "width":     1328,
        "height":    1328,
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

    def _load(self) -> _AnyQwenPipe:
        device = self._get_device()

        if self.gguf_file:
            pipe = self._load_gguf(device)
        else:
            pipe = self._load_standard(device)

        return pipe

    def _load_standard(self, device: torch.device) -> _AnyQwenPipe:
        """Load a standard (non-GGUF) Qwen-Image pipeline via from_pretrained.

        On MPS with ≤16 GB unified memory, loads in float16 with sequential
        CPU offload to stay within memory limits (20B model = ~40 GB bfloat16).
        """
        from diffusers import QwenImagePipeline  # type: ignore[attr-defined]
        # bfloat16 is recommended; fall back to float32 on CPU.
        # Qwen-Image is a 20B model (~40 GB in bfloat16 / ~20 GB in float16).
        # On MPS with ≤16 GB unified memory we must use float16 and CPU offload.
        # float16 is preferred over bfloat16 on MPS: MPS has limited bfloat16
        # support and silently upcasts some ops to float32, doubling memory use.
        if self.sequential_cpu_offload and device.type == "mps":
            # Load directly onto CPU in float16 so we never spike past RAM limit.
            # enable_sequential_cpu_offload() will move each layer to MPS just-
            # in-time during the forward pass.
            dtype = torch.float16
            self._log("⚙️  Low-memory mode: loading in float16 on CPU for sequential MPS offload.")
        elif device.type == "mps":
            dtype = torch.float16  # float16 is safer than bfloat16 on MPS
        elif device.type == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self._log(f"Loading Qwen-Image model: {self.model_id}  dtype={dtype} ...")
        pipe = QwenImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

        if self.sequential_cpu_offload:
            if device.type == "mps":
                self._log("⚙️  Sequential CPU offload enabled (MPS low-memory mode).")
                pipe.enable_sequential_cpu_offload()
            else:
                self._log("⚙️  Model CPU offload enabled (CUDA low-VRAM mode).")
                pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        self._apply_lora(pipe)
        return pipe

    def _load_gguf(self, device: torch.device) -> _AnyQwenPipe:
        """Load a GGUF-quantised Qwen-Image transformer and compose a full pipeline.

        The GGUF file contains only the transformer weights.  All other
        components (VAE, text encoder, scheduler, tokenizer) are loaded
        from components_repo (e.g. Qwen/Qwen-Image-2512).

        Q4_0 (~11.9 GB) fits within 16 GB unified memory without offloading.
        sequential_cpu_offload is incompatible with GGUF quantised tensors.
        """
        from diffusers import QwenImagePipeline  # type: ignore[attr-defined]
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

        if not self.base_model_id:
            raise ValueError(
                "GGUF loading requires 'components_repo' to be set in the config. "
                "For Qwen-Image-2512 GGUF set: "
                '"components_repo": "Qwen/Qwen-Image-2512"'
            )

        # bfloat16 is the compute dtype for GGUF dequantisation.
        # On MPS, float16 is safer (MPS has limited bfloat16 support).
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

        transformer = self._load_gguf_transformer(QwenImageTransformer2DModel, dtype)
        self._log(f"Loading pipeline components from: {self.base_model_id} ...")
        pipe = QwenImagePipeline.from_pretrained(
            self.base_model_id,
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

        pipe = self._apply_cpu_offload(pipe, device)
        self._apply_lora(pipe)
        return pipe
