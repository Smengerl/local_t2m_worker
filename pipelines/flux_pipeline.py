"""
FLUX pipeline backend — covers all FLUX model variants in a single class.

Dispatches to the correct diffusers pipeline class based on pipeline_type:
  - pipeline_type "flux2_klein"  →  Flux2KleinPipeline  (4B, distilled, guidance=1.0)
  - everything else               →  FluxPipeline        (FLUX.1-schnell / dev, 12B)

FLUX.1 additional features:
  - GGUF quantised transformers (set gguf_file + base_model_id in config)
  - LoRA weights (lora_id + optional weight_name)
  - T5 context length override (max_sequence_length)
  - guidance_scale MUST be 0.0 for FLUX.1-schnell (CFG-distilled)
  - guidance_scale 3.5–7.0 recommended for FLUX.1-dev and fine-tunes

FLUX.2 [klein] notes:
  - guidance_scale=1.0 per model card (NOT 0.0 like FLUX.1-schnell)
  - No LoRA, no GGUF, no T5 context override
  - Requires diffusers installed from git main (Flux2KleinPipeline not yet
    in a stable release as of April 2026):
      pip install git+https://github.com/huggingface/diffusers.git
"""

import os
import warnings
from typing import Any, Callable, Optional

import torch
from PIL import Image

from pipeline_config import PipelineConfig
from pipelines.base import BasePipeline

# Imported lazily inside _load() to avoid pulling in diffusers at module
# import time (keeps test / CLI startup fast when the pipeline is not used).
_AnyFluxPipe = Any  # FluxPipeline | Flux2KleinPipeline


class FluxBackend(BasePipeline):
    """Diffusers-based backend for all FLUX model variants.

    Handles FLUX.1 (schnell/dev, optional GGUF + LoRA) and
    FLUX.2 [klein] (4B distilled) behind a unified interface.
    """

    # FLUX.1-schnell is CFG-distilled → guidance_scale must be 0.0, few steps suffice.
    # FLUX.1-dev / fine-tunes should set cfg_scale in their config (3.5–7.0).
    GENERATION_DEFAULTS = {
        "steps":     4,
        "cfg_scale": 0.0,
        "width":     1024,
        "height":    1024,
    }

    # T5 context length default: 256 for schnell, 512 recommended for dev.
    # Overridden per-config via max_sequence_length in the JSON config.
    _DEFAULT_MAX_SEQUENCE_LENGTH: int = 256

    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg)
        self._is_klein: bool = cfg.pipeline_type == "flux2_klein"
        self._max_seq_len: int = cfg.max_sequence_length or self._DEFAULT_MAX_SEQUENCE_LENGTH
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
            num_inference_steps=total,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
        )

        # T5 context-length override is a FLUX.1-only parameter.
        # FLUX.2 [klein] does not expose max_sequence_length.
        if not self._is_klein:
            kwargs["max_sequence_length"] = self._max_seq_len

        if progress_callback is not None:
            def _cb(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
                progress_callback(step_index + 1, total)
                return callback_kwargs
            kwargs["callback_on_step_end"] = _cb

        result = self._pipe(**kwargs)
        return result.images[0]  # type: ignore[index]

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self) -> Any:
        device = self._get_device()

        # bfloat16 is the recommended dtype for all FLUX variants on both CUDA
        # and MPS. It avoids NaN instabilities that float16 can cause in the
        # FLUX transformer family.
        dtype = torch.bfloat16

        # Suppress deprecation warning emitted by older diffusers internals that
        # still pass local_dir_use_symlinks to hf_hub_download. The argument has
        # been a no-op since huggingface_hub ≥0.23 and will be removed in a
        # future release. The warning originates inside the diffusers library,
        # not in our code, so suppressing it here avoids noise until diffusers
        # drops the argument on their side.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*local_dir_use_symlinks.*",
                category=UserWarning,
            )
            if self._is_klein:
                pipe = self._load_klein(device, dtype)
            elif self.gguf_file:
                pipe = self._load_gguf(device, dtype)
            else:
                pipe = self._load_flux1(device, dtype)

        # ── memory optimisations (shared by all variants) ─────────────────
        if self.sequential_cpu_offload and not self.gguf_file:
            # On MPS (Apple Silicon), enable_model_cpu_offload() is CUDA-only
            # and silently falls back to CPU. Use enable_sequential_cpu_offload()
            # instead — it is MPS-compatible and moves submodules to CPU between
            # ops, keeping peak unified-memory use within ~16 GB.
            # NOTE: enable_sequential_cpu_offload() is incompatible with GGUF-
            # quantised transformers (accelerate tries to move GGUF tensors to
            # the meta device, but quant_type is None → KeyError). GGUF models
            # are already compact enough to fit in memory without offloading.
            if device.type == "mps" or not torch.cuda.is_available():
                self._log("⚙️  Sequential CPU offload enabled (MPS-compatible mode).")
                pipe.enable_sequential_cpu_offload()
            else:
                self._log("⚙️  Model CPU offload enabled (CUDA low-VRAM mode).")
                pipe.enable_model_cpu_offload()
        elif self.sequential_cpu_offload and self.gguf_file:
            self._log("⚙️  sequential_cpu_offload ignored for GGUF model (incompatible with GGUF quantised tensors). GGUF transformer fits in memory without offloading.")
            pipe = pipe.to(device)
        else:
            pipe = pipe.to(device)

        pipe.enable_vae_slicing()       # decode large images in slices → less peak RAM
        pipe.enable_vae_tiling()        # tile VAE for very large resolutions
        pipe.enable_attention_slicing() # slice attention heads one at a time → reduces peak
                                        # attention memory significantly; critical on 16 GB MPS

        # ── LoRA (FLUX.1 only) ────────────────────────────────────────────
        if self.lora_id:
            if self._is_klein:
                self._log("⚠️  LoRA is not supported for FLUX.2 [klein] — lora_id ignored.")
            else:
                self._log(f"Loading LoRA weights: {self.lora_id}  (scale={self.lora_scale}) ...")
                lora_kwargs: dict = {"cache_dir": self.cache_dir}
                if self.weight_name:
                    lora_kwargs["weight_name"] = self.weight_name
                    self._log(f"  Using specific weight file: {self.weight_name}")
                elif os.environ.get("HF_HUB_OFFLINE", "0") not in ("", "0"):
                    raise ValueError(
                        f"Cannot load LoRA '{self.lora_id}' in offline mode without a "
                        f"'weight_name'. Add \"weight_name\": \"<filename>.safetensors\" "
                        f"to your config file."
                    )
                pipe.load_lora_weights(self.lora_id, **lora_kwargs)
                if self.gguf_file:
                    # fuse_lora() merges LoRA deltas into the base weights via
                    # tensor addition. This is incompatible with GGUF quantised
                    # tensors (quantized blocks don't support in-place float add
                    # → RuntimeError: size mismatch at non-singleton dimension).
                    # Use set_adapters() instead: the LoRA adapter is kept
                    # separate and applied dynamically during each forward pass.
                    # diffusers auto-names the first adapter "default_0".
                    loaded_adapters = getattr(pipe, "peft_config", {})
                    adapter_name = list(loaded_adapters.keys())[0] if loaded_adapters else "default_0"
                    pipe.set_adapters([adapter_name], adapter_weights=[self.lora_scale])
                    self._log(f"LoRA weights loaded (unfused, dynamic application via set_adapters — GGUF mode).")
                else:
                    pipe.fuse_lora(lora_scale=self.lora_scale)
                    self._log("LoRA weights fused successfully.")

        return pipe

    def _load_flux1(self, device: torch.device, dtype: torch.dtype) -> Any:
        """Load a standard (non-GGUF) FLUX.1 pipeline."""
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

        self._log(f"Loading FLUX.1 model: {self.model_id} ...")
        return FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

    def _load_klein(self, device: torch.device, dtype: torch.dtype) -> Any:
        """Load the FLUX.2 [klein] 4B distilled pipeline.

        Requires diffusers from git main (Flux2KleinPipeline not yet released):
            pip install git+https://github.com/huggingface/diffusers.git
        """
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline  # type: ignore[import]

        self._log(f"Loading FLUX.2 [klein] model: {self.model_id} ...")
        return Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

    def _load_gguf(self, device: torch.device, dtype: torch.dtype) -> Any:
        """Load a GGUF-quantised FLUX.1 transformer and compose a full pipeline.

        The GGUF file contains only the transformer weights; all other
        components (text encoders, VAE, scheduler, tokenizers) are loaded
        from ``base_model_id`` (e.g. black-forest-labs/FLUX.1-dev).

        This follows the diffusers GGUF loading pattern:
            transformer = FluxTransformer2DModel.from_single_file(gguf_path, ...)
            pipe = FluxPipeline.from_pretrained(base_model_id, transformer=transformer)
        """
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
        from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        from diffusers import GGUFQuantizationConfig  # type: ignore[attr-defined]

        if not self.base_model_id:
            raise ValueError(
                "GGUF loading requires 'base_model_id' to be set in the config. "
                "For FLUX.1-dev based models set: \"base_model_id\": \"black-forest-labs/FLUX.1-dev\""
            )

        # Pass the HuggingFace URL directly so diffusers detects the .gguf
        # extension and routes to the correct GGUF loader.  Using a local
        # path from hf_hub_download can fail if the cached filename does not
        # retain the .gguf suffix (hash-based symlink layouts in newer hf-hub).
        gguf_url = f"https://huggingface.co/{self.model_id}/blob/main/{self.gguf_file}"
        self._log(f"Loading GGUF transformer from: {gguf_url} ...")
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_url,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        self._log(f"Loading pipeline components from base model: {self.base_model_id} ...")
        return FluxPipeline.from_pretrained(
            self.base_model_id,
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )
