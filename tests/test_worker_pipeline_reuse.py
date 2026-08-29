"""Finding 5 — the worker must not evict the pipeline cache before every job.

The old loop compared _cached_model (which held cfg.backend, e.g. "sd")
against job["pipeline_config"]["model"]["repo"] (a repo path).  They never
matched, so _release_pipeline_cache() ran before every job and the pipeline
was reloaded from scratch every time — defeating the documented reuse.
"""

import asyncio

import pytest

import batch.worker as w
import batch.queue as q
from pipeline_config import PipelineConfig


def test_derive_cached_model_uses_repo_not_backend():
    key = PipelineConfig.from_json("configs/sd15_default.json").pipeline_cache_key()
    assert key[0] == "sd"                       # backend
    assert w._derive_cached_model({key: object()}) == key[1]
    assert key[1] != "sd"
    assert w._derive_cached_model({}) is None


@pytest.fixture
def queue_env(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "QUEUE_FILE", tmp_path / "queue.jsonl")
    monkeypatch.setattr(q, "_LOCK_FILE", tmp_path / "queue.jsonl.lock")
    monkeypatch.setattr(w, "_cached_model", None, raising=False)
    return tmp_path


def _enqueue(repo):
    cfg = PipelineConfig.from_json("configs/sd15_default.json")
    cfg.apply_overrides(model_repo=repo)
    return q.enqueue(cfg=cfg, prompt="p")


def test_pipeline_reused_across_same_config_jobs(queue_env, monkeypatch):
    loads: list[str] = []

    def fake_process_job(job, pipeline_cache):
        cfg = PipelineConfig.from_dict(job["pipeline_config"])
        key = cfg.pipeline_cache_key()
        if key not in pipeline_cache:
            pipeline_cache.clear()          # mimic generate_image eviction on miss
            pipeline_cache[key] = object()
            loads.append(key[1])
        return f"outputs/{job['id']}.png"

    monkeypatch.setattr(w, "process_job", fake_process_job)

    _enqueue("model/a")
    _enqueue("model/a")
    _enqueue("model/b")

    asyncio.run(w.run_worker_async(keep_alive=False))

    # A loaded once (2nd A job reused it), B loaded once — NOT four loads
    assert loads == ["model/a", "model/b"]
    assert all(q.get_job(j["id"])["status"] == "done" for j in q.list_jobs())
