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
