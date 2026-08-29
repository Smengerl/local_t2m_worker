"""Finding 9 — generation.seed must actually reach the pipeline call.

It was parsed into cfg.seed / self.seed and then read by nobody.  Now
_build_generate_kwargs turns it into a torch.Generator.
"""

import torch

from pipelines.base import BasePipeline
from pipelines.flux_pipeline import FluxBackend
from pipelines.qwen_pipeline import QwenImageBackend
from pipelines.lumina2_pipeline import Lumina2Backend


class _FakeSelf:
    guidance_scale = 7.5
    width = 512
    height = 512
    seed = None
    # attrs the subclass overrides also touch
    _is_klein = False
    _max_seq_len = 256
    true_cfg_scale = None


def _kwargs(cls, seed):
    # build an instance without running __init__ (which would load a model)
    s = object.__new__(cls) if cls is not BasePipeline else _FakeSelf()
    for k, v in vars(_FakeSelf).items():
        if not k.startswith("__"):
            setattr(s, k, v)
    s.seed = seed
    return cls._build_generate_kwargs(s, "a cat", "", 10)


def test_no_seed_means_no_generator():
    assert "generator" not in _kwargs(BasePipeline, None)


def test_seed_becomes_generator_with_that_seed():
    g = _kwargs(BasePipeline, 12345)["generator"]
    assert isinstance(g, torch.Generator)
    assert g.initial_seed() == 12345


def test_seed_zero_is_honoured():
    g = _kwargs(BasePipeline, 0)["generator"]
    assert isinstance(g, torch.Generator)
    assert g.initial_seed() == 0


def test_generator_is_deterministic():
    g1 = torch.Generator(device="cpu").manual_seed(7)
    g2 = torch.Generator(device="cpu").manual_seed(7)
    assert torch.equal(torch.randn(8, generator=g1), torch.randn(8, generator=g2))


def test_subclasses_pass_generator_through():
    for cls in (FluxBackend, QwenImageBackend, Lumina2Backend):
        kw = _kwargs(cls, 99)
        assert kw["generator"].initial_seed() == 99, cls.__name__
