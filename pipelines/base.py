"""
Abstract base class for all image generation pipeline backends.

Every concrete pipeline must:
  - Accept a PipelineConfig in its constructor
  - Implement generate() returning a PIL Image
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Optional

import torch
from PIL import Image

from pipeline_config import PipelineConfig

log = logging.getLogger("pipelines")


class BasePipeline(ABC):
    """Common interface for all text-to-image pipeline backends.

    Subclasses must define ``GENERATION_DEFAULTS`` with backend-appropriate
    fallback values for all four numeric generation parameters.  These defaults
    are applied in ``__init__`` whenever the corresponding field in the config
    is ``None`` (i.e. neither the config file nor a CLI override specified it).
    """

    # Subclasses override this with their own sensible values.
    GENERATION_DEFAULTS: ClassVar[dict[str, Any]] = {
        "steps":     30,
        "cfg_scale": 7.5,
        "width":     1024,
        "height":    1024,
    }

    def __init__(self, cfg: PipelineConfig) -> None:
        d = self.GENERATION_DEFAULTS
        self.pipeline_type = cfg.pipeline_type
        self.model_id = cfg.model_id
        self.cache_dir = cfg.cache_dir
        # Resolve None → backend default
        self.num_inference_steps: int = cfg.num_inference_steps if cfg.num_inference_steps is not None else d["steps"]
        self.guidance_scale: float    = cfg.guidance_scale      if cfg.guidance_scale      is not None else d["cfg_scale"]
        self.width: int               = cfg.width               if cfg.width               is not None else d["width"]
        self.height: int              = cfg.height              if cfg.height              is not None else d["height"]
        self.lora_id = cfg.lora_id
        self.lora_scale = cfg.lora_scale
        self.sequential_cpu_offload = cfg.sequential_cpu_offload
        self.true_cfg_scale = cfg.true_cfg_scale
        self.seed = cfg.seed
        self.weight_name: Optional[str] = cfg.weight_name
        self.gguf_file: Optional[str] = cfg.gguf_file
        self.base_model_id: Optional[str] = cfg.base_model_id

    # ── shared GGUF / LoRA helpers ────────────────────────────────────────────

    def _build_gguf_url(self) -> str:
        """Return the HuggingFace HTTPS URL for the configured GGUF file.

        Diffusers detects the ``.gguf`` extension in the URL and routes to the
        correct GGUF loader automatically.  Using the URL (rather than a local
        cache path) avoids hash-based symlink layouts that may strip the suffix.
        """
        return f"https://huggingface.co/{self.model_id}/blob/main/{self.gguf_file}"

    def _load_gguf_transformer(
        self,
        transformer_cls: type,
        dtype: torch.dtype,
    ) -> Any:
        """Load a GGUF-quantised transformer via ``transformer_cls.from_single_file``.

        Builds the HuggingFace URL from ``self.model_id`` + ``self.gguf_file``,
        then calls ``from_single_file`` with a ``GGUFQuantizationConfig``.

        Args:
            transformer_cls: Diffusers transformer class to instantiate
                (e.g. ``FluxTransformer2DModel``, ``ZImageTransformer2DModel``).
            dtype: Compute dtype for GGUF dequantisation (``bfloat16`` or
                ``float16``).

        Returns:
            Loaded transformer model ready to be injected into a pipeline via
            ``Pipeline.from_pretrained(..., transformer=transformer)``.
        """
        from diffusers import GGUFQuantizationConfig  # type: ignore[attr-defined]

        gguf_url = self._build_gguf_url()
        self._log(f"Loading GGUF transformer from: {gguf_url} ...")
        return transformer_cls.from_single_file(
            gguf_url,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )

    def _apply_cpu_offload(self, pipe: Any, device: torch.device) -> Any:
        """Apply memory-offloading strategy and move *pipe* to *device*.

        Three cases:
        - ``cpu_offload`` enabled, no GGUF → ``enable_sequential_cpu_offload``
          (MPS) or ``enable_model_cpu_offload`` (CUDA).
        - ``cpu_offload`` enabled, GGUF present → log warning and move to
          device.  CPU offload is incompatible with GGUF quantised tensors
          (accelerate tries to move tensors to the meta device, but
          ``quant_type`` is ``None`` → ``KeyError``).  GGUF models are compact
          enough to fit in memory without offloading.
        - ``cpu_offload`` disabled → move to device directly.

        Args:
            pipe: Loaded diffusers pipeline.
            device: Target compute device.

        Returns:
            The pipeline (with offload hooks, or already on *device*).
        """
        if self.sequential_cpu_offload and not self.gguf_file:
            if device.type == "mps" or not torch.cuda.is_available():
                self._log("⚙️  Sequential CPU offload enabled (MPS-compatible mode).")
                pipe.enable_sequential_cpu_offload()
            else:
                self._log("⚙️  Model CPU offload enabled (CUDA low-VRAM mode).")
                pipe.enable_model_cpu_offload()
        elif self.sequential_cpu_offload and self.gguf_file:
            self._log(
                "⚙️  sequential_cpu_offload ignored for GGUF model (incompatible with "
                "GGUF quantised tensors). GGUF transformer fits in memory without offloading."
            )
            pipe = pipe.to(device)
        else:
            pipe = pipe.to(device)
        return pipe

    def _apply_lora(self, pipe: Any) -> None:
        """Load LoRA weights and apply them to *pipe*.

        Dispatches between ``fuse_lora()`` (standard float models) and
        ``set_adapters()`` (GGUF models, where ``fuse_lora()`` is incompatible
        with quantised tensors — it tries to add float deltas in-place to
        blocked quantised weights).

        Does nothing if ``self.lora_id`` is ``None``.

        Args:
            pipe: Loaded diffusers pipeline that supports ``load_lora_weights``.

        Raises:
            ValueError: If the LoRA's target modules do not exist in the base
                model's architecture (incompatible model family).
        """
        if not self.lora_id:
            return
        self._log(f"Loading LoRA weights: {self.lora_id}  (scale={self.lora_scale}) ...")
        lora_kwargs: dict = {"cache_dir": self.cache_dir}
        if self.weight_name:
            lora_kwargs["weight_name"] = self.weight_name
            self._log(f"  Using weight file: {self.weight_name}")
        try:
            pipe.load_lora_weights(self.lora_id, **lora_kwargs)
            if self.gguf_file:
                # fuse_lora() merges LoRA deltas into base weights via tensor
                # addition — incompatible with GGUF quantised blocks.  Use
                # set_adapters() to apply the adapter dynamically each forward
                # pass.  diffusers auto-names the first loaded adapter "default_0".
                loaded_adapters = getattr(pipe, "peft_config", {})
                adapter_name = list(loaded_adapters.keys())[0] if loaded_adapters else "default_0"
                pipe.set_adapters([adapter_name], adapter_weights=[self.lora_scale])
                self._log("LoRA weights loaded (unfused, dynamic via set_adapters — GGUF mode).")
            else:
                pipe.fuse_lora(lora_scale=self.lora_scale)
                self._log("LoRA weights fused successfully.")
        except ValueError as exc:
            raise ValueError(
                f"LoRA '{self.lora_id}' is incompatible with base model '{self.model_id}'.\n"
                f"The LoRA's target modules do not exist in the base model's architecture.\n"
                f"Original error: {exc}"
            ) from exc

    def _log(self, msg: str) -> None:
        """Emit *msg* at INFO level.

        Routes through ``logging.getLogger("pipelines")`` so that — when
        running inside the batch worker — the message is captured by the
        per-job ``_JobLogHandler`` and appears in the web-UI console.
        During direct CLI use the root logging configuration forwards it to
        stdout as usual.
        """
        log.info(msg)

    def _get_device(self) -> torch.device:
        """Return the best available compute device."""
        if torch.cuda.is_available():
            self._log("Using device: CUDA")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            self._log("Using device: MPS (Apple Silicon)")
            return torch.device("mps")
        self._log("Using device: CPU (this will be slow)")
        return torch.device("cpu")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Image.Image:
        """Run inference and return a PIL Image.

        Args:
            prompt: Text description of the image to generate.
            negative_prompt: Things to avoid in the image.
            progress_callback: Optional callable(step, total_steps) invoked
                after each denoising step. Used by the batch worker to write
                live progress into the queue. May be None (direct CLI use).

        Returns:
            Generated PIL Image.
        """
        ...
