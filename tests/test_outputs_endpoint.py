"""Finding 2 — /api/outputs containment check must compare path components.

The old check used ``str.startswith(str(OUTPUTS_DIR))``, so a sibling
directory whose name merely starts with ``outputs`` (e.g. ``outputs-backup``)
passed and its files could be served or deleted.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    sibling = tmp_path / "outputs-backup"
    sibling.mkdir()

    (outputs / "img.png").write_bytes(b"PNG")
    (outputs / "sub").mkdir()
    (outputs / "sub" / "nested.png").write_bytes(b"PNG")
    (sibling / "secret.png").write_bytes(b"SECRET")
    (tmp_path / "queue.jsonl").write_text("[]")

    from batch.api import outputs as outputs_mod

    monkeypatch.setattr(outputs_mod, "OUTPUTS_DIR", outputs)
    # keep PROJECT_ROOT consistent with the patched OUTPUTS_DIR so that
    # project-root-relative paths ("outputs/x.png") resolve as in production
    monkeypatch.setattr("batch.paths.PROJECT_ROOT", tmp_path)

    app = FastAPI()
    app.include_router(outputs_mod.router, prefix="/api")
    return TestClient(app), tmp_path


def _get(tc, abspath):
    # leading "//" so the {filename:path} param keeps its absolute leading slash
    return tc.get("/api/outputs/" + str(abspath))


def _delete(tc, abspath):
    return tc.delete("/api/outputs/" + str(abspath))


def test_serves_file_inside_outputs(client):
    tc, tmp = client
    r = _get(tc, tmp / "outputs" / "img.png")
    assert r.status_code == 200
    assert r.content == b"PNG"


def test_serves_nested_file(client):
    tc, tmp = client
    assert _get(tc, tmp / "outputs" / "sub" / "nested.png").status_code == 200


def test_serves_project_root_relative_path(client):
    # the frontend sends result_path as "outputs/<file>.png" (relative to root)
    tc, _ = client
    r = tc.get("/api/outputs/outputs/img.png")
    assert r.status_code == 200


def test_rejects_outputs_prefixed_sibling(client):
    tc, tmp = client
    r = _get(tc, tmp / "outputs-backup" / "secret.png")
    assert r.status_code == 403


def test_rejects_traversal_to_project_file(client):
    tc, tmp = client
    r = _get(tc, tmp / "outputs" / ".." / "queue.jsonl")
    assert r.status_code == 403


def test_delete_removes_file(client):
    tc, tmp = client
    target = tmp / "outputs" / "img.png"
    assert _delete(tc, target).status_code == 200
    assert not target.exists()


def test_delete_rejects_sibling(client):
    tc, tmp = client
    target = tmp / "outputs-backup" / "secret.png"
    assert _delete(tc, target).status_code == 403
    assert target.exists()


def test_delete_directory_is_404_not_500(client):
    tc, tmp = client
    assert _delete(tc, tmp / "outputs" / "sub").status_code == 404
