"""
Abstract base class for all image generation pipeline backends.

Every concrete pipeline must:
  - Accept a PipelineConfig in its constructor
  - Implement generate() returning a PIL Image
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from PIL import Image

from pipeline_config import PipelineConfig

log = logging.getLogger("pipelines")


class BasePipeline(ABC):
    """Common interface for all text-to-image pipeline backends.

    Subclasses receive a single ``PipelineConfig`` instance.  All config
    values are unpacked into named member variables so pipeline code can
    reference ``self.model_id``, ``self.num_inference_steps``, etc. as before.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.pipeline_type = cfg.pipeline_type
        self.model_id = cfg.model_id
        self.cache_dir = cfg.cache_dir
        self.num_inference_steps = cfg.num_inference_steps
        self.guidance_scale = cfg.guidance_scale
        self.width = cfg.width
        self.height = cfg.height
        self.lora_id = cfg.lora_id
        self.lora_scale = cfg.lora_scale
        self.sequential_cpu_offload = cfg.sequential_cpu_offload
        self.adapter_id = cfg.adapter_id
        self.true_cfg_scale = cfg.true_cfg_scale
        self.seed = cfg.seed
        self.weight_name: Optional[str] = getattr(cfg, "weight_name", None)

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
