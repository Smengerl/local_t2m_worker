"""Opt-5 — FLUX.2 [klein] must default guidance_scale to 1.0, not 0.0.

GENERATION_DEFAULTS["cfg_scale"] is 0.0 (correct for FLUX.1-schnell, which is
CFG-distilled). klein keeps a small positive CFG; 0.0 gives degraded output.
"""

from pipelines.flux_pipeline import FluxBackend
from pipeline_config import PipelineConfig


def _apply(is_klein, cfg_scale):
    s = object.__new__(FluxBackend)
    s._is_klein = is_klein
    raw = {"backend": "flux2_klein" if is_klein else "flux", "model": {"repo": "x"}}
    if cfg_scale is not None:
        raw["generation"] = {"cfg_scale": cfg_scale}
    cfg = PipelineConfig.from_dict(raw)
    assert (cfg.guidance_scale is None) == (cfg_scale is None)  # guard the fixture
    s.apply_generation_params(cfg)
    return s.guidance_scale


def test_klein_without_cfg_defaults_to_one():
    assert _apply(is_klein=True, cfg_scale=None) == 1.0


def test_klein_explicit_cfg_wins():
    assert _apply(is_klein=True, cfg_scale=3.5) == 3.5


def test_non_klein_keeps_schnell_default():
    assert _apply(is_klein=False, cfg_scale=None) == 0.0
