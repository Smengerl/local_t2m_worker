"""Opt-6 — dtype choices per pipeline/device, and no revival of the
"bfloat16 is broken on MPS" myth.

Research + an on-device check (torch 2.11 / Apple Silicon) showed bf16 works
fine on MPS and does not upcast or double memory.  flux/lumina2/zimage
correctly use bf16 on MPS; qwen uses fp16 there purely for the 20B memory
budget.  These tests pin that so a future "let's make it consistent" pass
doesn't silently flip zimage/flux to fp16.
"""

import pathlib

import pytest
import torch

_PIPE_DIR = pathlib.Path(__file__).resolve().parent.parent / "pipelines"

_MYTH_PHRASES = [
    "limited bfloat16 support",
    "silently upcasts",
    "MPS has limited bfloat16",
]


@pytest.mark.parametrize("path", sorted(_PIPE_DIR.glob("*.py")), ids=lambda p: p.name)
def test_no_bfloat16_mps_myth_in_comments(path):
    text = path.read_text()
    for phrase in _MYTH_PHRASES:
        assert phrase not in text, f"{path.name}: stale bf16/MPS myth: {phrase!r}"


def _capture_dtype(monkeypatch, backend_cls, *, gguf: bool, device_type: str):
    seen = {}

    def fake_sub_load(self, device, dtype, *a, **kw):
        seen["dtype"] = dtype
        m = pytest.importorskip("unittest.mock")
        return m.MagicMock()

    for name in ("_load_standard", "_load_gguf", "_load_flux1", "_load_klein"):
        if hasattr(backend_cls, name):
            monkeypatch.setattr(backend_cls, name, fake_sub_load, raising=False)
    monkeypatch.setattr(backend_cls, "_apply_cpu_offload", lambda self, pipe, dev: pipe)
    monkeypatch.setattr(backend_cls, "_apply_lora", lambda self, pipe: None, raising=False)
    monkeypatch.setattr(backend_cls, "_get_device", lambda self: torch.device(device_type))

    s = object.__new__(backend_cls)
    s.gguf_file = "m.gguf" if gguf else None
    s._is_klein = False
    s.model_id = "x"
    s.cache_dir = None
    s.sequential_cpu_offload = False
    s.lora_id = None
    s.lora_scale = None
    s.weight_name = None
    s._load()
    return seen["dtype"]


@pytest.mark.parametrize("device_type", ["mps", "cuda"])
def test_zimage_uses_bfloat16_on_gpu(monkeypatch, device_type):
    from pipelines.zimage_pipeline import ZImageBackend
    assert _capture_dtype(monkeypatch, ZImageBackend, gguf=False, device_type=device_type) is torch.bfloat16


def test_zimage_cpu_is_float32(monkeypatch):
    from pipelines.zimage_pipeline import ZImageBackend
    assert _capture_dtype(monkeypatch, ZImageBackend, gguf=False, device_type="cpu") is torch.float32


@pytest.mark.parametrize("device_type", ["mps", "cuda"])
def test_flux_uses_bfloat16_on_gpu(monkeypatch, device_type):
    from pipelines.flux_pipeline import FluxBackend
    assert _capture_dtype(monkeypatch, FluxBackend, gguf=False, device_type=device_type) is torch.bfloat16
