"""Finding 3 — queue.jsonl must be rewritten atomically.

The old _write_all truncated then rewrote the file in place, so a crash
mid-write left a partial last line that made every later _read_all() raise.
"""

import json

import pytest

import batch.queue as q


@pytest.fixture
def queue_file(tmp_path, monkeypatch):
    qf = tmp_path / "queue.jsonl"
    monkeypatch.setattr(q, "QUEUE_FILE", qf)
    monkeypatch.setattr(q, "_LOCK_FILE", tmp_path / "queue.jsonl.lock")
    return qf


JOBS = [{"id": "a", "status": "pending"}, {"id": "b", "status": "done"}]


def test_round_trip(queue_file):
    q._write_all(JOBS)
    assert q._read_all() == JOBS
    assert not queue_file.with_suffix(".jsonl.tmp").exists()


def test_replace_failure_keeps_old_file_intact(queue_file, monkeypatch):
    q._write_all(JOBS)                       # establish a good file
    original = queue_file.read_text()

    monkeypatch.setattr(q.os, "replace", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        q._write_all([{"id": "c", "status": "pending"}])

    assert queue_file.read_text() == original      # untouched, still complete
    assert q._read_all() == JOBS                    # and still parseable
    assert not queue_file.with_suffix(".jsonl.tmp").exists()   # tmp cleaned up


def test_serialization_failure_does_not_touch_queue_file(queue_file):
    q._write_all(JOBS)
    original = queue_file.read_text()

    class Bad:
        def __repr__(self):  # json.dumps will fail on this
            raise ValueError("unserializable")

    with pytest.raises((TypeError, ValueError)):
        q._write_all([{"id": "c"}, {"bad": Bad()}])

    assert queue_file.read_text() == original
    assert not queue_file.with_suffix(".jsonl.tmp").exists()


def test_read_all_reports_corrupt_line(queue_file):
    queue_file.write_text('{"id": "a", "status": "pending"}\n{"id": "b", tr')
    with pytest.raises(ValueError, match="corrupt at line 2"):
        q._read_all()


def test_public_api_round_trips(queue_file):
    from pipeline_config import PipelineConfig

    cfg = PipelineConfig.from_json("configs/sd15_default.json")
    job = q.enqueue(cfg=cfg, prompt="a cat")
    assert q.get_job(job["id"])["prompt"] == "a cat"
    assert json.loads(queue_file.read_text().splitlines()[0])["id"] == job["id"]
