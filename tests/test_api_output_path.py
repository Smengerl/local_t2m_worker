"""Finding 1 — POST /api/jobs must reject an `output` path outside outputs/.

The worker writes the generated PNG to ``job["output"]`` verbatim, so an
unrestricted value is an arbitrary-file-overwrite primitive.  Also exercises
``batch.paths.is_within``, which replaces the buggy ``str.startswith``
containment check (finding 2).
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from batch.paths import OUTPUTS_DIR, PROJECT_ROOT, is_within


# ── is_within ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "candidate, expected",
    [
        (OUTPUTS_DIR / "a.png", True),
        (OUTPUTS_DIR / "sub" / "a.png", True),
        (OUTPUTS_DIR, True),
        (str(OUTPUTS_DIR) + "-backup/secret.png", False),  # the startswith bug
        (PROJECT_ROOT / "queue.jsonl", False),
        (OUTPUTS_DIR / ".." / "etc" / "passwd", False),
        ("/etc/passwd", False),
    ],
)
def test_is_within(candidate, expected):
    assert is_within(candidate, OUTPUTS_DIR) is expected


# ── POST /api/jobs ───────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    """A TestClient over just the jobs router, with enqueue/notify stubbed."""
    from batch.api import jobs

    captured = {}

    def fake_enqueue(**kwargs):
        captured.update(kwargs)
        return {"id": "test", "status": "pending", "output": kwargs.get("output")}

    monkeypatch.setattr(jobs, "enqueue", fake_enqueue)
    monkeypatch.setattr(jobs.notify, "notify", lambda: None)
    monkeypatch.setattr(
        jobs.PipelineConfig, "from_json",
        classmethod(lambda cls, path: _StubCfg()),
    )

    app = FastAPI()
    app.include_router(jobs.router, prefix="/api")
    return TestClient(app), captured


class _StubCfg:
    trigger_word = ""

    def apply_overrides(self, **kw):
        pass

    def any_trigger_in_prompt(self, prompt):
        return True


def test_rejects_output_outside_outputs(client):
    tc, _ = client
    resp = tc.post("/api/jobs", json={
        "config": "configs/sd15_default.json",
        "prompt": "x",
        "output": "/tmp/evil.png",
    })
    assert resp.status_code == 400


def test_rejects_outputs_prefixed_sibling(client):
    tc, _ = client
    resp = tc.post("/api/jobs", json={
        "config": "configs/sd15_default.json",
        "prompt": "x",
        "output": str(OUTPUTS_DIR) + "-backup/x.png",
    })
    assert resp.status_code == 400


def test_accepts_output_inside_outputs(client):
    tc, captured = client
    resp = tc.post("/api/jobs", json={
        "config": "configs/sd15_default.json",
        "prompt": "x",
        "output": str(OUTPUTS_DIR / "ok.png"),
    })
    assert resp.status_code == 200
    assert captured["output"] == str(OUTPUTS_DIR / "ok.png")


def test_accepts_no_output(client):
    tc, _ = client
    resp = tc.post("/api/jobs", json={
        "config": "configs/sd15_default.json",
        "prompt": "x",
    })
    assert resp.status_code == 200
