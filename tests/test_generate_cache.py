"""Finding 6 — generate_image must free the old model before loading the new.

Cache miss ordering: evict (clear + empty_cache) has to happen *before*
create_pipeline(), or both models are briefly resident and a 16 GB Mac
OOM-kills the process.  Also covers that a cache *hit* does not reload.
"""

import pytest

import generate as g
from pipeline_config import PipelineConfig


@pytest.fixture
def spy(monkeypatch, tmp_path):
    events: list[str] = []
    params: list[tuple] = []          # (width, steps) as seen at generate() time

    class FakePipe:
        def __init__(self, key):
            self.key = key
            self.width = self.num_inference_steps = None

        def apply_generation_params(self, cfg):
            self.width = cfg.width
            self.num_inference_steps = cfg.num_inference_steps

        def generate(self, prompt, negative_prompt, progress_callback=None):
            events.append(f"generate:{self.key}")
            params.append((self.width, self.num_inference_steps))
            return _FakeImage()

    class _FakeImage:
        def save(self, path):
            pass

    def fake_create(cfg):
        key = cfg.pipeline_cache_key()
        events.append(f"create:{key[1]}")
        return FakePipe(key[1])

    monkeypatch.setattr(g, "create_pipeline", fake_create)

    # make eviction observable without touching torch
    real_evict = g._evict_pipeline_cache

    def spy_evict(cache):
        if cache:
            events.append("evict")
        real_evict(cache)

    monkeypatch.setattr(g, "_evict_pipeline_cache", spy_evict)

    def run(cfg):
        return g.generate_image(
            cfg, str(tmp_path / "o.png"), "p", "", pipeline_cache=cache
        )

    cache: dict = {}
    return run, events, cache, params


def _cfg(repo, *, width=None, steps=None):
    c = PipelineConfig.from_json("configs/sd15_default.json")
    c.apply_overrides(model_repo=repo, width=width, steps=steps)
    return c


def test_miss_evicts_before_create(spy):
    run, events, cache, _ = spy
    run(_cfg("model/a"))
    run(_cfg("model/b"))

    assert events == [
        "create:model/a",
        "generate:model/a",
        "evict",            # <-- before the second create
        "create:model/b",
        "generate:model/b",
    ]
    assert set(cache) == {_cfg("model/b").pipeline_cache_key()}


def test_hit_does_not_reload(spy):
    run, events, cache, _ = spy
    run(_cfg("model/a"))
    run(_cfg("model/a"))

    assert events == ["create:model/a", "generate:model/a", "generate:model/a"]
    assert "evict" not in events


def test_reused_pipeline_gets_fresh_sampling_params(spy):
    run, events, cache, params = spy
    run(_cfg("model/a", width=512, steps=10))
    run(_cfg("model/a", width=1024, steps=40))   # same cache key, different size/steps

    assert events.count("create:model/a") == 1   # pipeline reused
    assert params == [(512, 10), (1024, 40)]     # but each render used its own params


def test_no_cache_never_evicts(spy):
    run, events, cache, _ = spy
    # pipeline_cache=None path
    g.generate_image(_cfg("model/a"), "/dev/null", "p", "", pipeline_cache=None)
    assert events == ["create:model/a", "generate:model/a"]
