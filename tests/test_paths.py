"""Unit tests for batch.paths — the shared containment helpers."""

import pytest

from batch.paths import PROJECT_ROOT, is_within, resolve_within


def test_is_within_true_cases(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    assert is_within(base, base)
    assert is_within(base / "a.txt", base)
    assert is_within(base / "x" / "y.txt", base)


def test_is_within_rejects_prefix_sibling(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "base-backup").mkdir()
    assert not is_within(tmp_path / "base-backup" / "f.txt", base)


def test_is_within_rejects_traversal(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    assert not is_within(base / ".." / "other.txt", base)


def test_resolve_within_project_root_relative():
    # "configs/..." is stored relative to the project root, not to CONFIGS_DIR
    got = resolve_within("pyproject.toml", PROJECT_ROOT)
    assert got == (PROJECT_ROOT / "pyproject.toml")


def test_resolve_within_escape_raises(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError):
        resolve_within("/etc/passwd", base)


def test_resolve_within_missing_raises(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_within("nope.txt", base)


def test_resolve_within_missing_ok_when_not_required(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    got = resolve_within("nope.txt", base, must_exist=False)
    assert got == base / "nope.txt"
