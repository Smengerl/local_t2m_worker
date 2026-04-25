#!/usr/bin/env python3
"""
Pre-download HuggingFace model weights for one or more pipeline configs.

Uses ``snapshot_download()`` / ``hf_hub_download()`` directly so that:
  - Already-cached blobs are detected and skipped automatically.
  - Partially-downloaded blobs (`.incomplete`) are resumed, not re-started.
  - The local cache structure is identical to what ``from_pretrained()`` expects,
    so the pipeline loads from cache on the next run without any network calls.

Usage
-----
  # Download everything needed by one config
  python scripts/preload_model.py -c configs/flux_schnell.json

  # Download all configs at once
  python scripts/preload_model.py -c configs/flux_schnell.json configs/sdxl_graffiti_lora.json

  # Download to a custom cache dir
  python scripts/preload_model.py -c configs/flux_schnell.json --cache-dir models/

  # Dry-run: only list what would be downloaded
  python scripts/preload_model.py -c configs/flux_schnell.json --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: re-exec with the project venv Python if we're running under the
# system interpreter (which won't have huggingface_hub, torch, etc.)
# ---------------------------------------------------------------------------
_VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _read_token(token_path: Optional[str]) -> Optional[str]:
    """Load a HuggingFace token from file (same logic as the pipeline)."""
    candidates = [token_path, ".hf_token", str(Path.home() / ".huggingface" / "token")]
    for p in candidates:
        if p and Path(p).is_file():
            tok = Path(p).read_text().strip()
            if tok:
                return tok
    return None


def _repo_ids_from_config(cfg: dict) -> list[tuple[str, str]]:
    """Extract all HuggingFace repo IDs from a merged config dict.

    Returns a list of (repo_id, role) tuples where *role* is the config key
    that referenced the repo (e.g. ``"model_id"``, ``"base_model_id"``, …).
    """
    keys = ["model_id", "adapter_id", "lora_id", "base_model_id"]
    results = []
    for k in keys:
        val = cfg.get(k)
        if val and isinstance(val, str) and not Path(val).exists():
            results.append((val, k))
    return results


def _load_config(path: str) -> dict:
    """Load a JSON config, stripping ``_comment`` keys."""
    with open(path) as fh:
        raw = json.load(fh)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _is_gguf_repo(cfg: dict) -> bool:
    return bool(cfg.get("gguf_file"))


# ── per-repo download ─────────────────────────────────────────────────────────

def _download_repo(
    repo_id: str,
    *,
    cache_dir: Optional[str],
    token: Optional[str],
    dry_run: bool,
    ignore_patterns: Optional[list[str]] = None,
) -> None:
    """Download (or verify) all blobs of *repo_id* via snapshot_download."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    log.info("──────────────────────────────────────────")
    log.info("Repo : %s", repo_id)
    log.info("Cache: %s", cache_dir or "~/.cache/huggingface (default)")

    if dry_run:
        log.info("[dry-run] Would call snapshot_download(%s)", repo_id)
        if ignore_patterns:
            log.info("[dry-run] ignore_patterns=%s", ignore_patterns)
        return

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir or None,
            token=token,
            ignore_patterns=ignore_patterns or [],
            # huggingface_hub ≥0.12: incomplete blobs are always resumed.
        )
        log.info("✔  Cached at: %s", local_dir)
    except RepositoryNotFoundError:
        log.error("✖  Repository not found: %s", repo_id)
        log.error("   Check the model ID or accept the license on huggingface.co")
    except EntryNotFoundError as exc:
        log.error("✖  Entry not found in %s: %s", repo_id, exc)
    except Exception as exc:
        log.error("✖  Download failed for %s: %s", repo_id, exc)
        raise


def _download_single_file(
    repo_id: str,
    filename: str,
    *,
    cache_dir: Optional[str],
    token: Optional[str],
    dry_run: bool,
) -> None:
    """Download a single file (e.g. a .gguf weight) from *repo_id*."""
    from huggingface_hub import hf_hub_download

    log.info("──────────────────────────────────────────")
    log.info("Repo : %s", repo_id)
    log.info("File : %s", filename)
    log.info("Cache: %s", cache_dir or "~/.cache/huggingface (default)")

    if dry_run:
        log.info("[dry-run] Would call hf_hub_download(%s, %s)", repo_id, filename)
        return

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir or None,
            token=token,
        )
        log.info("✔  Cached at: %s", local)
    except Exception as exc:
        log.error("✖  Download failed (%s / %s): %s", repo_id, filename, exc)
        raise


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download HuggingFace model weights (resume-safe).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--config",
        metavar="FILE",
        nargs="+",
        required=True,
        help="One or more JSON config files whose models should be downloaded.",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        default=None,
        help=(
            "Cache directory for downloaded weights. "
            "Defaults to the value in the config, then to the HuggingFace default "
            "(~/.cache/huggingface)."
        ),
    )
    parser.add_argument(
        "--token-file",
        metavar="FILE",
        default=None,
        help="Path to HF token file (default: .hf_token in project root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    # Load token once
    token = _read_token(args.token_file)
    if token:
        log.info("HF token loaded.")
    else:
        log.info("No HF token found — gated models will be inaccessible.")

    # Collect all (repo_id, role, gguf_file, ignore_patterns, cache_dir) tuples
    tasks: list[tuple[str, str, Optional[str], list[str], Optional[str]]] = []
    seen: set[str] = set()

    for config_path in args.config:
        cfg = _load_config(config_path)
        effective_cache = args.cache_dir or cfg.get("cache_dir")

        for repo_id, role in _repo_ids_from_config(cfg):
            key = f"{repo_id}|{role}|{effective_cache}"
            if key in seen:
                continue
            seen.add(key)

            gguf_file: Optional[str] = None
            ignore: list[str] = []

            if role == "model_id" and _is_gguf_repo(cfg):
                # GGUF transformer repo: only fetch the single quantised file,
                # then the lightweight metadata (no other large tensors here).
                gguf_file = cfg.get("gguf_file")

            elif role == "model_id" and not _is_gguf_repo(cfg):
                # Full diffusers pipeline repo (non-GGUF).
                # Many repos additionally ship GGUF quantised variants at the
                # repository root (e.g. FHDR_ComfyUI-Q4_K_M.gguf = 6.93 GB,
                # FHDR_ComfyUI-Q8_0.gguf = 12.7 GB).
                # from_pretrained() never reads these — it uses model_index.json
                # and only loads files inside the component subdirectories.
                # Excluding *.gguf saves tens of GB without affecting the
                # downloaded diffusers pipeline.
                #
                # NOTE on ComfyUI single-file safetensors (e.g.
                # FHDR_ComfyUI.safetensors = 23.8 GB): these also sit at
                # the repo root and are never used by from_pretrained(), but
                # Python's fnmatch does NOT treat '/' as a path boundary, so
                # "*.safetensors" would also match component shards inside
                # transformer/, vae/, text_encoder_2/ — those MUST be kept.
                # To exclude a known root-level single-file use a config-level
                # preload_ignore_patterns entry (not yet implemented).
                ignore = ["*.gguf"]
                log.info(
                    "Full pipeline repo: skipping *.gguf files "
                    "(GGUF variants not used by from_pretrained)."
                )

            elif role == "base_model_id" and _is_gguf_repo(cfg):
                # Base pipeline repo used only for VAE, text encoders, tokenizer
                # and scheduler — the transformer comes from the GGUF file.
                # Skip the full-precision transformer weights to avoid
                # downloading tens of GB that will never be used.
                # Mirrors what diffusers does at runtime: when transformer= is
                # already passed to from_pretrained(), it excludes transformer/
                # from allow_patterns automatically (see pipeline_utils.py).
                ignore = ["transformer/*"]
                log.info(
                    "GGUF base repo: skipping transformer/ weights "
                    "(transformer comes from gguf_file). Only VAE, "
                    "text encoders and tokenizer will be downloaded."
                )

            elif role == "lora_id":
                weight_name: Optional[str] = cfg.get("weight_name")
                if weight_name:
                    # load_lora_weights() calls hf_hub_download for exactly
                    # this one file when weight_name is set — so we mirror that
                    # behaviour instead of snapshot-downloading the whole repo.
                    gguf_file = weight_name   # reuse single-file download path
                    log.info(
                        "LoRA repo with weight_name — will download only: %s",
                        weight_name,
                    )
                # else: no weight_name → diffusers auto-selects the file via
                # snapshot, so we snapshot the whole repo (correct).

            tasks.append((repo_id, role, gguf_file, ignore, effective_cache))

    if not tasks:
        log.warning("No HuggingFace repo IDs found in the provided configs.")
        sys.exit(0)

    log.info("%d model repo(s) to process.", len(tasks))
    errors = 0

    for repo_id, role, gguf_file, ignore, cache_dir in tasks:
        try:
            if gguf_file and role == "lora_id":
                # LoRA with explicit weight_name: only the single .safetensors
                # file is needed — load_lora_weights() uses hf_hub_download
                # for exactly this file. No snapshot of the rest required.
                _download_single_file(
                    repo_id,
                    gguf_file,
                    cache_dir=cache_dir,
                    token=token,
                    dry_run=args.dry_run,
                )
            elif gguf_file:
                # GGUF transformer: download the quantised weight file first,
                # then snapshot the remaining repo metadata (config, etc.)
                # while skipping all other .gguf variants.
                _download_single_file(
                    repo_id,
                    gguf_file,
                    cache_dir=cache_dir,
                    token=token,
                    dry_run=args.dry_run,
                )
                _download_repo(
                    repo_id,
                    cache_dir=cache_dir,
                    token=token,
                    dry_run=args.dry_run,
                    ignore_patterns=["*.gguf"],
                )
            else:
                _download_repo(
                    repo_id,
                    cache_dir=cache_dir,
                    token=token,
                    dry_run=args.dry_run,
                    ignore_patterns=ignore,
                )
        except Exception as exc:
            log.error("✖  Unexpected error for %s: %s", repo_id, exc, exc_info=True)
            errors += 1

    log.info("──────────────────────────────────────────")
    if errors:
        log.error("%d repo(s) failed. Check errors above.", errors)
        sys.exit(1)
    else:
        log.info("All models ready. Use --offline flag in run.sh / run_batch_server.sh to skip network checks.")


if __name__ == "__main__":
    main()
