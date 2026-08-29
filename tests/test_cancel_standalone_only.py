"""Finding 4 — `batch.cancel` must not signal the web server's PID.

When the worker runs embedded in `batch.server`, the job's worker_pid is the
server PID.  The CLI used to os.kill() it unconditionally, taking the whole
server (and every queued job) down.  It now only signals a PID it can confirm
belongs to a standalone worker via batch/worker.pid.
"""

import signal

import pytest

import batch.cancel as cancel
import batch.queue as q


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "QUEUE_FILE", tmp_path / "queue.jsonl")
    monkeypatch.setattr(q, "_LOCK_FILE", tmp_path / "queue.jsonl.lock")
    monkeypatch.setattr(cancel, "WORKER_PID_FILE", tmp_path / "worker.pid")

    alive: set[int] = set()
    signals: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        if sig == 0:
            if pid not in alive:
                raise ProcessLookupError(pid)
            return
        signals.append((pid, sig))
        alive.discard(pid)  # pretend the target exits on the signal

    monkeypatch.setattr(cancel.os, "kill", fake_kill)
    return tmp_path, alive, signals


def _running_job(worker_pid: int):
    from pipeline_config import PipelineConfig
    cfg = PipelineConfig.from_json("configs/sd15_default.json")
    job = q.enqueue(cfg=cfg, prompt="x")
    q.update_job(job["id"], status="running", worker_pid=worker_pid)
    return job["id"]


def test_refuses_when_pid_is_not_the_standalone_worker(env):
    tmp, alive, signals = env
    server_pid = 4242
    alive.add(server_pid)                       # server process is alive
    # no worker.pid file -> not a standalone worker
    job_id = _running_job(server_pid)

    rc = cancel._cancel_job(job_id)

    assert rc == 1
    assert signals == []                        # never signalled
    assert q.get_job(job_id)["status"] == "running"   # untouched


def test_signals_confirmed_standalone_worker(env):
    tmp, alive, signals = env
    worker_pid = 5252
    alive.add(worker_pid)
    (tmp / "worker.pid").write_text(str(worker_pid))
    job_id = _running_job(worker_pid)

    rc = cancel._cancel_job(job_id)

    assert rc == 0
    assert signals == [(worker_pid, signal.SIGTERM)]
    assert q.get_job(job_id)["status"] == "failed"


def test_force_sends_sigkill_to_standalone_worker(env):
    tmp, alive, signals = env
    worker_pid = 6363
    alive.add(worker_pid)
    (tmp / "worker.pid").write_text(str(worker_pid))
    job_id = _running_job(worker_pid)

    rc = cancel._cancel_job(job_id, force=True)

    assert rc == 0
    assert signals == [(worker_pid, signal.SIGKILL)]


def test_dead_worker_pid_is_just_cleaned_up(env):
    tmp, alive, signals = env
    job_id = _running_job(9999)                 # 9999 not in `alive`

    rc = cancel._cancel_job(job_id)

    assert rc == 0
    assert signals == []
    assert q.get_job(job_id)["status"] == "failed"


def test_stale_pidfile_does_not_authorise_signalling(env):
    tmp, alive, signals = env
    job_pid = 7474
    alive.add(job_pid)
    (tmp / "worker.pid").write_text("1111")     # stale: 1111 is not alive
    job_id = _running_job(job_pid)

    rc = cancel._cancel_job(job_id)

    assert rc == 1
    assert signals == []
