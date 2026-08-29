"""Opt-4 — GGUF transformer load must honour system.cache_dir / --offline.

_load_gguf_transformer handed from_single_file a huggingface.co URL and no
cache_dir, so a preloaded .gguf in a custom cache was re-downloaded into
~/.cache/huggingface and --offline broke for GGUF configs.
"""

import torch

from pipelines.base import BasePipeline


class _FakeSelf:
    model_id = "owner/model"
    gguf_file = "model-Q4_K_M.gguf"
    cache_dir = None

    def _log(self, *_a):
        pass


class _FakeTransformerCls:
    last = {}

    @classmethod
    def from_single_file(cls, path, **kwargs):
        cls.last = {"path": path, **kwargs}
        return "TRANSFORMER"


def _run(monkeypatch, cache_dir):
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return f"/hf/snapshots/main/{kwargs['filename']}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    s = _FakeSelf()
    s.cache_dir = cache_dir
    out = BasePipeline._load_gguf_transformer(s, _FakeTransformerCls, torch.bfloat16)
    return out, calls


def test_passes_cache_dir_to_download(monkeypatch):
    out, calls = _run(monkeypatch, "/custom/models")
    assert out == "TRANSFORMER"
    assert calls["cache_dir"] == "/custom/models"
    assert calls["repo_id"] == "owner/model"
    assert calls["filename"] == "model-Q4_K_M.gguf"


def test_from_single_file_gets_local_path_not_url(monkeypatch):
    _run(monkeypatch, "/custom/models")
    path = _FakeTransformerCls.last["path"]
    assert not path.startswith("http")
    assert path.endswith(".gguf")


def test_none_cache_dir_passes_none(monkeypatch):
    _out, calls = _run(monkeypatch, None)
    assert calls["cache_dir"] is None
