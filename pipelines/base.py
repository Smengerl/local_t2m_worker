"""
Abstract base class for all image generation pipeline backends.

Every concrete pipeline must:
  - Accept the full config dict in its constructor
  - Implement generate() returning a PIL Image
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

from PIL import Image


class BasePipeline(ABC):
    """Common interface for all text-to-image pipeline backends.

    Subclasses receive the fully resolved config dict on construction and
    perform all setup (model loading, weight downloads, device placement)
    inside __init__ or lazily on the first generate() call.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

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
