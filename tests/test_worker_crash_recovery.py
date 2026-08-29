"""Finding 8 — a crashed embedded worker must not leave jobs stuck 'running'.

- the worker loop marks its in-flight job failed and records the error
- the server heals in-process 'running' jobs when the loop is down
- the server supervisor restarts a crashing worker (bounded), then degrades
"""

import asyncio
import os

import pytest

import batch.worker as w
import batch.queue as q
import batch.server as s
from pipeline_config import PipelineConfig


@pytest.fixture
def queue_env(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "QUEUE_FILE", tmp_path / "queue.jsonl")
    monkeypatch.setattr(q, "_LOCK_FILE", tmp_path / "queue.jsonl.lock")
    return tmp_path


def _enqueue():
    cfg = PipelineConfig.from_json("configs/sd15_default.json")
    return q.enqueue(cfg=cfg, prompt="p")


# ── worker loop crash ────────────────────────────────────────────────────────

def test_loop_crash_marks_inflight_job_failed(queue_env, monkeypatch):
    job = _enqueue()

    def fake_process_job(j, cache):
        w._current_job_id = j["id"]          # simulate work started
        return "outputs/x.png"

    def boom_finish(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(w, "process_job", fake_process_job)
    monkeypatch.setattr(w, "_finish_job", boom_finish)

    with pytest.raises(RuntimeError):
        asyncio.run(w.run_worker_async(keep_alive=False))

    assert w.worker_is_alive() is False
    assert w._last_error and "kaboom" in w._last_error
    assert q.get_job(job["id"])["status"] == "failed"


# ── server heals in-process stale jobs ───────────────────────────────────────

def test_heal_marks_embedded_job_failed_when_loop_down(queue_env, monkeypatch):
    from batch.api import jobs

    job = _enqueue()
    q.update_job(job["id"], status="running", worker_pid=os.getpid())

    monkeypatch.setattr(w, "worker_is_alive", lambda: False)
    jobs._heal_stale_running_jobs()
    assert q.get_job(job["id"])["status"] == "failed"


def test_heal_leaves_embedded_job_running_when_loop_alive(queue_env, monkeypatch):
    from batch.api import jobs

    job = _enqueue()
    q.update_job(job["id"], status="running", worker_pid=os.getpid())

    monkeypatch.setattr(w, "worker_is_alive", lambda: True)
    jobs._heal_stale_running_jobs()
    assert q.get_job(job["id"])["status"] == "running"


# ── supervisor ───────────────────────────────────────────────────────────────

@pytest.fixture
def no_backoff(monkeypatch):
    async def _nosleep(*_a, **_kw):
        return
    monkeypatch.setattr(s.asyncio, "sleep", _nosleep)


def test_supervisor_restarts_then_succeeds(no_backoff, monkeypatch):
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("crash")
        return

    monkeypatch.setattr(s._worker, "run_worker_async", lambda **kw: flaky())
    asyncio.run(s._worker_supervisor())
    assert calls["n"] == 3            # 1 initial + 2 restarts


def test_supervisor_gives_up_after_max_restarts(no_backoff, monkeypatch):
    calls = {"n": 0}

    async def always_crash():
        calls["n"] += 1
        raise RuntimeError("crash")

    monkeypatch.setattr(s._worker, "run_worker_async", lambda **kw: always_crash())
    asyncio.run(s._worker_supervisor())            # returns, does not raise
    assert calls["n"] == s._MAX_WORKER_RESTARTS + 1


def test_supervisor_propagates_cancellation(no_backoff, monkeypatch):
    async def cancelled():
        raise asyncio.CancelledError

    monkeypatch.setattr(s._worker, "run_worker_async", lambda **kw: cancelled())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(s._worker_supervisor())
