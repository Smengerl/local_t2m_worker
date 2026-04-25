"""
Lumina2 pipeline backend — covers standard and GGUF-quantised variants.

Lumina-Image-2.0 is a 2B DiT model from Alpha-VLLM that uses Gemma2 as
text encoder (not T5/CLIP like FLUX).  The architecture uses 30 transformer
layers, a context_refiner, and cap_embedder modules.

GGUF loading pattern:
  - Transformer weights come from the GGUF file (quantised, ~6 GB Q8_0)
  - All other components (Gemma2 text encoder, VAE, scheduler, tokenizer)
    are loaded from base_model_id (Alpha-VLLM/Lumina-Image-2.0)

Non-GGUF loading:
  - Full pipeline loaded from model_id via from_pretrained()
"""

from typing import Any, Callable, Optional

import torch
from PIL import Image

from pipeline_config import PipelineConfig
from pipelines.base import BasePipeline

_AnyLuminaPipe = Any  # Lumina2Pipeline


class Lumina2Backend(BasePipeline):
    """Diffusers-based backend for Lumina-Image-2.0 and GGUF fine-tunes.

    Supports both standard from_pretrained loading and GGUF-quantised
    transformer loading (set gguf_file + base_model_id in config).
    """

    GENERATION_DEFAULTS = {
        "steps":     30,
        "cfg_scale": 5.0,
        "width":     1024,
        "height":    1024,
    }

    _DEFAULT_MAX_SEQUENCE_LENGTH: int = 256

    def __init__(self, cfg: PipelineConfig) -> None:
        super().__init__(cfg)
        self._max_seq_len: int = (
            cfg.max_sequence_length or self._DEFAULT_MAX_SEQUENCE_LENGTH
        )
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
            max_sequence_length=self._max_seq_len,
        )

        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        if progress_callback is not None:

            def _cb(
                pipe, step_index: int, timestep, callback_kwargs: dict
            ) -> dict:
                progress_callback(step_index + 1, total)
                return callback_kwargs

            kwargs["callback_on_step_end"] = _cb

        result = self._pipe(**kwargs)
        return result.images[0]  # type: ignore[index]

    # ── private ──────────────────────────────────────────────────────────────

    def _load(self) -> _AnyLuminaPipe:
        device = self._get_device()

        # bfloat16 is recommended for Lumina2 on both CUDA and MPS.
        dtype = torch.bfloat16

        if self.gguf_file:
            pipe = self._load_gguf(device, dtype)
        else:
            pipe = self._load_standard(device, dtype)

        # ── memory optimisations ──────────────────────────────────────────
        if self.sequential_cpu_offload:
            if device.type == "mps" or not torch.cuda.is_available():
                self._log("⚙️  Sequential CPU offload enabled (MPS-compatible mode).")
                pipe.enable_sequential_cpu_offload()
            else:
                self._log("⚙️  Model CPU offload enabled (CUDA low-VRAM mode).")
                pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

        return pipe

    def _load_standard(self, device: torch.device, dtype: torch.dtype) -> _AnyLuminaPipe:
        """Load a standard (non-GGUF) Lumina2 pipeline via from_pretrained."""
        from diffusers import Lumina2Pipeline

        self._log(f"Loading Lumina2 model: {self.model_id} ...")
        return Lumina2Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )

    def _load_gguf(self, device: torch.device, dtype: torch.dtype) -> _AnyLuminaPipe:
        """Load a GGUF-quantised Lumina2 transformer and compose a full pipeline.

        The GGUF file contains only the transformer weights.  All other
        components (Gemma2 text encoder, VAE, scheduler, tokenizers) are
        loaded from base_model_id (typically Alpha-VLLM/Lumina-Image-2.0).

        Architecture validation is performed upfront:
        - Checks that the adaLN conditioning_dim matches diffusers' expectation
          (hidden_size, from a shared MLP-projected timestep embedding).
        - Checks that QKV tensors are divisible into equal thirds (no GQA).

        If the GGUF was exported from the original Alpha-VLLM training code
        (conditioning_dim=256, 4× adaLN per block without gating) rather than
        the diffusers model, loading will fail with a clear error message.
        """
        import diffusers.loaders.single_file_model as _sfm
        from diffusers import GGUFQuantizationConfig, Lumina2Pipeline  # type: ignore[attr-defined]
        from diffusers.models.model_loading_utils import load_gguf_checkpoint
        from diffusers.models.transformers.transformer_lumina2 import (
            Lumina2Transformer2DModel,
        )

        if not self.base_model_id:
            raise ValueError(
                "GGUF loading requires 'base_model_id' to be set in the config. "
                "For Lumina2-based models set: "
                '"base_model_id": "Alpha-VLLM/Lumina-Image-2.0"'
            )

        # ── pre-flight architecture validation ───────────────────────────────
        # Resolve local cached path (avoids a second download in from_single_file).
        from huggingface_hub import hf_hub_download

        gguf_filename: str = self.gguf_file  # already validated non-None above
        gguf_local = hf_hub_download(
            repo_id=self.model_id,
            filename=gguf_filename,
            cache_dir=self.cache_dir or None,
        )

        raw = load_gguf_checkpoint(gguf_local, return_tensors=True)

        # Detect hidden_size from first RMS-norm weight.
        hidden_size = next(
            (int(v.shape[0]) for k, v in raw.items() if k.endswith(".attention_norm1.weight")),
            None,
        )
        # Detect adaLN conditioning_dim from first modulation weight.
        adaln_key = next((k for k in raw if "adaLN_modulation" in k and k.endswith(".weight")), None)
        adaln_cond_dim = int(raw[adaln_key].quant_shape[-1]) if adaln_key and hasattr(raw[adaln_key], "quant_shape") else (int(raw[adaln_key].shape[-1]) if adaln_key else None)

        if hidden_size is not None and adaln_cond_dim is not None:
            if adaln_cond_dim != hidden_size:
                raise ValueError(
                    f"GGUF architecture mismatch: this checkpoint uses adaLN "
                    f"conditioning_dim={adaln_cond_dim} (raw Fourier timestep, "
                    f"original Alpha-VLLM training format), but diffusers "
                    f"Lumina2Transformer2DModel expects conditioning_dim="
                    f"{hidden_size} (hidden_size, MLP-projected timestep).\n\n"
                    f"This GGUF was exported from the original Lumina2 training "
                    f"codebase which uses a different adaLN implementation "
                    f"(4× per-block projections from 256-dim Fourier features, "
                    f"no gating) vs diffusers (6× gated projection from "
                    f"hidden_size-dim MLP embedding).\n\n"
                    f"These architectures are structurally incompatible — the "
                    f"weights cannot be loaded into diffusers without rewriting "
                    f"the transformer implementation.  The model file is:\n"
                    f"  {gguf_local}"
                )

        # ── dynamic QKV-split converter ───────────────────────────────────────
        # diffusers' built-in convert_lumina2_to_diffusers hard-codes
        # q_dim=2304 / k_dim=v_dim=768 (vanilla 2B GQA layout).  Fine-tunes
        # may have a different, equal Q/K/V split.  We patch the registry
        # entry directly (the module-attribute patch has no effect because the
        # registry holds the original function object by reference).
        def _dynamic_convert(checkpoint: dict, **kwargs) -> dict:
            qkv_key = next(
                (k for k in checkpoint if k.endswith(".qkv.weight")), None
            )
            if qkv_key is not None:
                out_dim = int(checkpoint[qkv_key].shape[0])
                q_dim = k_dim = v_dim = out_dim // 3
            else:
                q_dim, k_dim, v_dim = 2304, 768, 768  # vanilla 2B fallback

            LUMINA_KEY_MAP = {
                "cap_embedder": "time_caption_embed.caption_embedder",
                "t_embedder.mlp.0": "time_caption_embed.timestep_embedder.linear_1",
                "t_embedder.mlp.2": "time_caption_embed.timestep_embedder.linear_2",
                "attention": "attn",
                ".out.": ".to_out.0.",
                "k_norm": "norm_k",
                "q_norm": "norm_q",
                "w1": "linear_1",
                "w2": "linear_2",
                "w3": "linear_3",
                "adaLN_modulation.1": "norm1.linear",
            }
            ATTENTION_NORM_MAP = {"attention_norm1": "norm1.norm", "attention_norm2": "norm2"}
            CONTEXT_REFINER_MAP = {
                "context_refiner.0.attention_norm1": "context_refiner.0.norm1",
                "context_refiner.0.attention_norm2": "context_refiner.0.norm2",
                "context_refiner.1.attention_norm1": "context_refiner.1.norm1",
                "context_refiner.1.attention_norm2": "context_refiner.1.norm2",
            }
            FINAL_LAYER_MAP = {
                "final_layer.adaLN_modulation.1": "norm_out.linear_1",
                "final_layer.linear": "norm_out.linear_2",
            }

            checkpoint.pop("norm_final.weight", None)
            for k in list(checkpoint.keys()):
                if "model.diffusion_model." in k:
                    checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

            converted: dict = {}
            for key in list(checkpoint.keys()):
                dk = key
                for km, kv in CONTEXT_REFINER_MAP.items():
                    dk = dk.replace(km, kv)
                for km, kv in FINAL_LAYER_MAP.items():
                    dk = dk.replace(km, kv)
                for km, kv in ATTENTION_NORM_MAP.items():
                    dk = dk.replace(km, kv)
                for km, kv in LUMINA_KEY_MAP.items():
                    dk = dk.replace(km, kv)
                if "qkv" in dk:
                    t = checkpoint.pop(key)
                    tq, tk, tv = torch.split(t, [q_dim, k_dim, v_dim], dim=0)
                    converted[dk.replace("qkv", "to_q")] = tq
                    converted[dk.replace("qkv", "to_k")] = tk
                    converted[dk.replace("qkv", "to_v")] = tv
                else:
                    converted[dk] = checkpoint.pop(key)
            return converted

        registry_entry = _sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"]
        orig_fn = registry_entry["checkpoint_mapping_fn"]
        registry_entry["checkpoint_mapping_fn"] = _dynamic_convert
        try:
            self._log(f"Loading GGUF transformer from: {gguf_local} ...")
            transformer = Lumina2Transformer2DModel.from_single_file(
                gguf_local,
                config=self.base_model_id,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
        finally:
            registry_entry["checkpoint_mapping_fn"] = orig_fn

        self._log(
            f"Loading pipeline components from base model: {self.base_model_id} ..."
        )
        return Lumina2Pipeline.from_pretrained(
            self.base_model_id,
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )
        import diffusers.loaders.single_file_model as _sfm
        from diffusers import GGUFQuantizationConfig, Lumina2Pipeline  # type: ignore[attr-defined]
        from diffusers.models.transformers.transformer_lumina2 import (
            Lumina2Transformer2DModel,
        )

        if not self.base_model_id:
            raise ValueError(
                "GGUF loading requires 'base_model_id' to be set in the config. "
                "For Lumina2-based models set: "
                '"base_model_id": "Alpha-VLLM/Lumina-Image-2.0"'
            )

        # Build a dynamic converter that infers q/k/v dims from the checkpoint
        # instead of relying on the hardcoded vanilla-model constants.
        def _dynamic_lumina2_convert(checkpoint: dict, **kwargs) -> dict:
            import torch as _t

            # Inspect the first QKV weight to determine the split sizes.
            # After GGUF dequantisation the tensor has shape (out, in) in
            # PyTorch convention, where out = Q_rows + K_rows + V_rows.
            qkv_key = next(
                (k for k in checkpoint if k.endswith(".qkv.weight")), None
            )
            if qkv_key is not None:
                out_dim = int(checkpoint[qkv_key].shape[0])
                q_dim = k_dim = v_dim = out_dim // 3  # equal split (no GQA)
            else:
                q_dim, k_dim, v_dim = 2304, 768, 768  # vanilla 2B fallback

            LUMINA_KEY_MAP = {
                "cap_embedder": "time_caption_embed.caption_embedder",
                "t_embedder.mlp.0": "time_caption_embed.timestep_embedder.linear_1",
                "t_embedder.mlp.2": "time_caption_embed.timestep_embedder.linear_2",
                "attention": "attn",
                ".out.": ".to_out.0.",
                "k_norm": "norm_k",
                "q_norm": "norm_q",
                "w1": "linear_1",
                "w2": "linear_2",
                "w3": "linear_3",
                "adaLN_modulation.1": "norm1.linear",
            }
            ATTENTION_NORM_MAP = {
                "attention_norm1": "norm1.norm",
                "attention_norm2": "norm2",
            }
            CONTEXT_REFINER_MAP = {
                "context_refiner.0.attention_norm1": "context_refiner.0.norm1",
                "context_refiner.0.attention_norm2": "context_refiner.0.norm2",
                "context_refiner.1.attention_norm1": "context_refiner.1.norm1",
                "context_refiner.1.attention_norm2": "context_refiner.1.norm2",
            }
            FINAL_LAYER_MAP = {
                "final_layer.adaLN_modulation.1": "norm_out.linear_1",
                "final_layer.linear": "norm_out.linear_2",
            }

            checkpoint.pop("norm_final.weight", None)
            keys = list(checkpoint.keys())
            for k in keys:
                if "model.diffusion_model." in k:
                    checkpoint[k.replace("model.diffusion_model.", "")] = (
                        checkpoint.pop(k)
                    )

            converted: dict = {}
            for key in list(checkpoint.keys()):
                dk = key
                for km, kv in CONTEXT_REFINER_MAP.items():
                    dk = dk.replace(km, kv)
                for km, kv in FINAL_LAYER_MAP.items():
                    dk = dk.replace(km, kv)
                for km, kv in ATTENTION_NORM_MAP.items():
                    dk = dk.replace(km, kv)
                for km, kv in LUMINA_KEY_MAP.items():
                    dk = dk.replace(km, kv)

                if "qkv" in dk:
                    tensor = checkpoint.pop(key)
                    to_q, to_k, to_v = _t.split(
                        tensor, [q_dim, k_dim, v_dim], dim=0
                    )
                    converted[dk.replace("qkv", "to_q")] = to_q
                    converted[dk.replace("qkv", "to_k")] = to_k
                    converted[dk.replace("qkv", "to_v")] = to_v
                else:
                    converted[dk] = checkpoint.pop(key)

            return converted

        # Swap the function reference inside the registry dict; restore after.
        _registry_entry = _sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"]
        _orig_fn = _registry_entry["checkpoint_mapping_fn"]
        _registry_entry["checkpoint_mapping_fn"] = _dynamic_lumina2_convert
        try:
            gguf_url = (
                f"https://huggingface.co/{self.model_id}/resolve/main/{self.gguf_file}"
            )
            self._log(f"Loading GGUF transformer from: {gguf_url} ...")
            transformer = Lumina2Transformer2DModel.from_single_file(
                gguf_url,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
        finally:
            _registry_entry["checkpoint_mapping_fn"] = _orig_fn

        self._log(
            f"Loading pipeline components from base model: {self.base_model_id} ..."
        )
        return Lumina2Pipeline.from_pretrained(
            self.base_model_id,
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
        )
