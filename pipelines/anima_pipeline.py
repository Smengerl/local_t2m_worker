"""
Anima / AnimaYume pipeline backend (Cosmos-Predict2 architecture).

Anima is a 2B-parameter anime text-to-image model by CircleStone Labs built on
NVIDIA Cosmos-Predict2. It uses a Qwen-3 LLM as text encoder and a Qwen Image
VAE — an architecture that diffusers does not support natively.

Inference runs via the **stable-diffusion.cpp** binary (sd / sd-cli).

Installation (macOS Apple Silicon):
  Download the pre-built ARM64 release from:
    https://github.com/leejet/stable-diffusion.cpp/releases/latest
    → pick the file ending in -bin-Darwin-macOS-*-arm64.zip
  Then:
    unzip sd-master-*-bin-Darwin-macOS-*-arm64.zip
    sudo mv sd /usr/local/bin/ && sudo chmod +x /usr/local/bin/sd

Required config keys:
  model_id        str   HF repo of the fine-tuned diffusion model
  pipeline_type   str   must be "anima"
  cache_dir       str   local directory for downloaded model weights
  lora_id       str|null
  lora_scale    float
  num_inference_steps int
  guidance_scale  float
  width           int
  height          int

Model files downloaded automatically on first run (~14 GB total):
  split_files/diffusion_models/<model>.safetensors  (diffusion backbone)
  split_files/text_encoders/qwen_3_06b_base.safetensors
  split_files/vae/qwen_image_vae.safetensors
"""

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

from pipelines.base import BasePipeline

_TEXT_ENCODER_FILE = "split_files/text_encoders/qwen_3_06b_base.safetensors"
_VAE_FILE = "split_files/vae/qwen_image_vae.safetensors"
_ANIMA_BASE_REPO = "circlestone-labs/Anima"


class AnimaPipeline(BasePipeline):
    """Cosmos-Predict2 / Anima backend via stable-diffusion.cpp subprocess."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._sd_bin = self._find_binary()
        self._diffusion, self._text_enc, self._vae = self._download_components(
            cfg["model_id"], cfg["cache_dir"]
        )

    # ── public ───────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,  # not used — subprocess
    ) -> Image.Image:
        cfg = self.cfg
        lora_id: Optional[str] = cfg.get("lora_id") or None
        lora_scale: float = float(cfg.get("lora_scale", 0.9))

        full_prompt = prompt
        lora_dir: Optional[str] = None
        if lora_id:
            lora_path = self._resolve_lora(lora_id, cfg["cache_dir"])
            full_prompt = f"<lora:{lora_path.stem}:{lora_scale}> {prompt}"
            lora_dir = str(lora_path.parent)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            out = tmp.name

        cmd: list[str] = [
            self._sd_bin,
            "--diffusion-model", str(self._diffusion),
            "--llm",             str(self._text_enc),
            "--vae",             str(self._vae),
            "-p",  full_prompt,
            "-n",  negative_prompt,
            "-W",  str(int(cfg.get("width", 1024))),
            "-H",  str(int(cfg.get("height", 1024))),
            "--steps",          str(int(cfg["num_inference_steps"])),
            "--cfg-scale",      str(float(cfg["guidance_scale"])),
            "--sampling-method", "euler",
            "--diffusion-fa",
            "--vae-tiling",
            "-o", out,
            "-v",
        ]
        seed = int(cfg.get("seed", -1))
        if seed >= 0:
            cmd += ["-s", str(seed)]
        if lora_dir:
            cmd += ["--lora-model-dir", lora_dir]

        print(f"Running sd binary: {shlex.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sd binary exited with code {result.returncode}.\n"
                "Check the output above for error details."
            )

        img = Image.open(out)
        img.load()
        return img

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_binary() -> str:
        for name in ("sd", "sd-cli"):
            binary = shutil.which(name)
            if binary:
                return binary
        raise RuntimeError(
            "stable-diffusion.cpp binary not found on PATH.\n\n"
            "Download the pre-built macOS ARM64 release (~20 MB) from:\n"
            "  https://github.com/leejet/stable-diffusion.cpp/releases/latest\n"
            "  → pick the file ending in  -bin-Darwin-macOS-*-arm64.zip\n\n"
            "Then install it:\n"
            "  unzip sd-master-*-bin-Darwin-macOS-*-arm64.zip\n"
            "  sudo mv sd /usr/local/bin/\n"
            "  sudo chmod +x /usr/local/bin/sd\n\n"
            "Alternatively, build from source:\n"
            "  https://github.com/leejet/stable-diffusion.cpp#build"
        )

    @staticmethod
    def _download_file(repo_id: str, filename: str, cache_dir: str) -> Path:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir))

    @classmethod
    def _download_components(
        cls, model_id: str, cache_dir: str
    ) -> tuple[Path, Path, Path]:
        print(f"Resolving Anima model components for: {model_id}")
        print(f"  (Shared text encoder + VAE from {_ANIMA_BASE_REPO})")

        prefix = "split_files/diffusion_models/"
        candidates = sorted(
            f for f in list_repo_files(model_id)
            if f.startswith(prefix) and f.endswith(".safetensors")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .safetensors found under {prefix!r} in repo {model_id!r}.\n"
                f"Check https://huggingface.co/{model_id}"
            )
        chosen = candidates[-1]
        print(f"  Diffusion model : {chosen}")

        diffusion = cls._download_file(model_id, chosen, cache_dir)
        text_enc = cls._download_file(_ANIMA_BASE_REPO, _TEXT_ENCODER_FILE, cache_dir)
        vae = cls._download_file(_ANIMA_BASE_REPO, _VAE_FILE, cache_dir)

        print(f"  Text encoder    : {text_enc.name}")
        print(f"  VAE             : {vae.name}")
        return diffusion, text_enc, vae

    @classmethod
    def _resolve_lora(cls, lora_id: str, cache_dir: str) -> Path:
        local = Path(lora_id)
        if local.exists():
            return local
        candidates = sorted(
            f for f in list_repo_files(lora_id)
            if "/" not in f and f.endswith(".safetensors")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No root-level .safetensors found in HF repo {lora_id!r}"
            )
        return cls._download_file(lora_id, candidates[-1], cache_dir)
