"""Opt-3 — a missing/gated repo must fail the preload, not pass silently.

_download_repo caught RepositoryNotFoundError / EntryNotFoundError, logged
them, and returned without raising, so main()'s error counter stayed 0 and
the script printed "All models ready" and exited 0.
"""

import pytest
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

import scripts.preload_model as pm


class _FakeRepoNotFound(RepositoryNotFoundError):
    def __init__(self):  # skip HfHubHTTPError.__init__ (needs a response)
        pass


class _FakeEntryNotFound(EntryNotFoundError):
    def __init__(self):
        pass


@pytest.mark.parametrize("exc", [_FakeRepoNotFound, _FakeEntryNotFound, RuntimeError])
def test_download_repo_propagates_failures(monkeypatch, exc):
    def boom(*a, **kw):
        raise exc() if exc is not RuntimeError else RuntimeError("net down")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    with pytest.raises((RepositoryNotFoundError, EntryNotFoundError, RuntimeError)):
        pm._download_repo("owner/nope", cache_dir=None, token=None, dry_run=False)


def test_download_single_file_propagates(monkeypatch):
    def boom(*a, **kw):
        raise _FakeRepoNotFound()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(RepositoryNotFoundError):
        pm._download_single_file("owner/nope", "x.gguf", cache_dir=None, token=None, dry_run=False)


def test_main_exits_nonzero_on_missing_repo(monkeypatch, tmp_path, caplog):
    cfg = tmp_path / "bad.json"
    cfg.write_text('{"backend": "sd", "model": {"repo": "owner/definitely-not-a-real-repo"}}')

    def boom(*a, **kw):
        raise _FakeRepoNotFound()

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    monkeypatch.setattr("sys.argv", ["preload_model.py", "-c", str(cfg)])

    with caplog.at_level("INFO"):
        with pytest.raises(SystemExit) as se:
            pm.main()

    assert se.value.code == 1
    assert "All models ready" not in caplog.text
