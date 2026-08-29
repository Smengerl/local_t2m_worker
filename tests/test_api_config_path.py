"""Finding 7 — POST /api/jobs must confine `config` to the configs/ directory.

An unrestricted path let a client read arbitrary files as JSON and turned a
missing/invalid path into an uncaught HTTP 500.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from batch.api import jobs

    calls = {"from_json": []}
    real_from_json = jobs.PipelineConfig.from_json.__func__

    def spy_from_json(cls, path):
        calls["from_json"].append(path)
        return real_from_json(cls, path)

    monkeypatch.setattr(jobs.PipelineConfig, "from_json", classmethod(spy_from_json))
    monkeypatch.setattr(jobs, "enqueue",
                        lambda **kw: {"id": "t", "status": "pending"})
    monkeypatch.setattr(jobs.notify, "notify", lambda: None)

    app = FastAPI()
    app.include_router(jobs.router, prefix="/api")
    return TestClient(app), calls


def _post(tc, config):
    return tc.post("/api/jobs", json={"config": config, "prompt": "a cat"})


def test_accepts_real_config(client):
    tc, calls = client
    assert _post(tc, "configs/sd15_default.json").status_code == 200
    assert calls["from_json"]


def test_rejects_path_outside_configs(client):
    tc, calls = client
    r = _post(tc, "../pyproject.toml")
    assert r.status_code == 400
    assert calls["from_json"] == []          # never reached the parser


def test_rejects_absolute_path(client):
    tc, _ = client
    assert _post(tc, "/etc/hosts").status_code == 400


def test_missing_config_is_404_not_500(client):
    tc, _ = client
    assert _post(tc, "configs/does_not_exist.json").status_code == 404


def test_default_config_still_works(client):
    tc, _ = client
    # EnqueueRequest.config defaults to configs/sd15_default.json
    assert tc.post("/api/jobs", json={"prompt": "a cat"}).status_code == 200
