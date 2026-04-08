"""
Stable Diffusion pipeline backend (SD 1.5, SDXL, and SD3).

Handles pipeline variants via pipeline_type or model_id heuristic:
  - pipeline_type "sd3"        →  StableDiffusion3Pipeline
  - model_id containing "xl"   →  StableDiffusionXLPipeline
  - everything else             →  StableDiffusionPipeline

Supports optional LoRA weights and memory-saving CPU offload mode.
"""

from typing import Callable, Optional

import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from PIL import Image

from pipelines.base import BasePipeline
from pipeline_config import PipelineConfig

_DiffusersPipe = StableDiffusionPipeline | StableDiffusionXLPipeline | StableDiffusion3Pipeline


class StableDiffusionBackend(BasePipeline):
    """Diffusers-based backend for SD 1.5 and Stable Diffusion XL."""

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
            negative_prompt=negative_prompt,
            num_inference_steps=total,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
        )

        if progress_callback is not None:
            def _cb(pipe, step_index: int, timestep, callback_kwargs: dict) -> dict:
                progress_callback(step_index + 1, total)  # 0-based → 1-based
                return callback_kwargs
            kwargs["callback_on_step_end"] = _cb

        result = self._pipe(**kwargs)
        return result.images[0]  # type: ignore[index]

    # ── private ──────────────────────────────────────────────────────────────

    def _is_sd3(self) -> bool:
        return self.pipeline_type == "sd3"

    @staticmethod
    def _is_xl(model_id: str) -> bool:
        return "xl" in model_id.lower()

    def _load(self) -> _DiffusersPipe:
        if self.adapter_id:
            self._log(f"⚠️  adapter_id={self.adapter_id!r} is set but adapter support is not yet implemented — ignored.")

        is_sd3 = self._is_sd3()
        is_xl = (not is_sd3) and self._is_xl(self.model_id)

        if is_sd3:
            pipeline_cls: type = StableDiffusion3Pipeline
            label = "SD3"
        elif is_xl:
            pipeline_cls = StableDiffusionXLPipeline
            label = "SDXL"
        else:
            pipeline_cls = StableDiffusionPipeline
            label = "SD"

        device = self._get_device()

        self._log(f"Loading {label} base model: {self.model_id} ...")

        # SD3 on MPS: float16 is best supported; float32 would OOM on 16 GB
        # SD/SDXL on MPS: float16 can produce NaN → use float32
        if device.type == "cuda":
            dtype = torch.float16
        elif is_sd3 and device.type == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32

        load_kwargs: dict = dict(torch_dtype=dtype, cache_dir=self.cache_dir)
        if not is_xl and not is_sd3:
            load_kwargs["safety_checker"] = None
            load_kwargs["requires_safety_checker"] = False

        if self.weight_name:
            # Single-file checkpoint — download to local cache first, then load
            # from the local path. from_single_file() with a URL internally calls
            # _get_model_file(repo_id, weights_name) which re-appends resolve/main/,
            # causing a double-path 404. Using hf_hub_download avoids this.
            self._log(f"Downloading single-file checkpoint: {self.model_id}/{self.weight_name} ...")
            local_path = hf_hub_download(
                repo_id=self.model_id,
                filename=self.weight_name,
                cache_dir=self.cache_dir,
            )
            self._log(f"Loading from local cache: {local_path} ...")
            pipe = pipeline_cls.from_single_file(local_path, **load_kwargs)
        else:
            pipe = pipeline_cls.from_pretrained(self.model_id, **load_kwargs)

        if self.sequential_cpu_offload:
            # Offloads submodules to CPU between ops; cuts peak memory ~50 %.
            # Must be called before .to(device).
            self._log("⚙️  Sequential CPU offload enabled (low-VRAM mode).")
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)

        pipe.enable_attention_slicing()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        if self.lora_id:
            self._log(f"Loading LoRA weights: {self.lora_id}  (scale={self.lora_scale}) ...")
            lora_kwargs: dict = {"cache_dir": self.cache_dir}
            if self.weight_name:
                lora_kwargs["weight_name"] = self.weight_name
                self._log(f"  Using specific weight file: {self.weight_name}")
            try:
                pipe.load_lora_weights(self.lora_id, **lora_kwargs)
            except RuntimeError as exc:
                if "size mismatch" not in str(exc):
                    raise
                # Legacy LoRA files (old diffusers) store proj_in/proj_out as
                # 4-D conv tensors [out, in, 1, 1] instead of 2-D linear [out, in].
                # Download, squeeze the extra dimensions, save to a temp file, then reload.
                self._log("⚠️  Legacy LoRA format detected (4-D conv weights). Patching and retrying ...")
                patched_path = self._download_and_patch_lora(self.lora_id)
                pipe.load_lora_weights(patched_path)
            pipe.fuse_lora(lora_scale=self.lora_scale)
            self._log("LoRA weights fused successfully.")

        return pipe

    def _download_and_patch_lora(self, lora_id: str) -> str:
        """Download a legacy LoRA (4-D conv proj weights) and return a path to a
        patched copy with squeezed 2-D linear weights that modern diffusers expects."""
        import tempfile
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors.torch import load_file, save_file

        # Find the first .safetensors file in the repo
        safetensors_files = [f for f in list_repo_files(lora_id) if f.endswith(".safetensors")]
        if not safetensors_files:
            raise RuntimeError(f"No .safetensors file found in LoRA repo: {lora_id}")
        weight_name = safetensors_files[0]

        local_path = hf_hub_download(
            repo_id=lora_id,
            filename=weight_name,
            cache_dir=self.cache_dir,
        )

        state_dict = load_file(local_path)
        patched: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            # Squeeze trailing 1x1 spatial dims: [out, in, 1, 1] → [out, in]
            if tensor.ndim == 4 and tensor.shape[-2:] == (1, 1):
                tensor = tensor.squeeze(-1).squeeze(-1)
            patched[key] = tensor

        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        save_file(patched, tmp.name)
        self._log(f"Patched LoRA saved to temp file: {tmp.name}")
        return tmp.name
