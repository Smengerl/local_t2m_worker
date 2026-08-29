"""Shared filesystem paths and path-safety helpers.

Centralises the ``PROJECT_ROOT`` / ``OUTPUTS_DIR`` constants that were
previously re-derived (as ``_ROOT`` / ``_OUTPUTS_DIR``) in half a dozen
modules, and provides a single correct directory-containment check.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"


def is_within(candidate: Path | str, base: Path | str) -> bool:
    """Return True if *candidate* resolves to a path inside *base*.

    Both paths are fully resolved (symlinks included) before comparison, so
    this is a real ancestor check — not a string prefix test.  ``base`` itself
    counts as inside ``base``.

    A plain ``str.startswith`` check is wrong here: with base
    ``/x/outputs`` it also accepts ``/x/outputs-backup/secret`` because that
    string starts with the base string.  ``Path.relative_to`` compares path
    components, so a sibling directory whose name merely shares a prefix is
    correctly rejected.
    """
    try:
        Path(candidate).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def resolve_within(raw: str, base: Path, *, must_exist: bool = True) -> Path:
    """Resolve *raw* and require the result to stay inside *base*.

    *raw* may be:
    - an absolute path (used as-is),
    - a path relative to the project root, e.g. ``"outputs/x.png"`` or
      ``"configs/y.json"`` (this is how ``result_path`` / config paths are
      stored),
    - a path relative to *base* itself, e.g. a bare ``"x.png"``.

    Raises ``ValueError`` if the resolved path escapes *base*, and
    ``FileNotFoundError`` if *must_exist* and the path does not exist.  Callers
    at an API boundary translate those into the appropriate HTTP status.
    """
    base = Path(base).resolve()
    p = Path(raw)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        from_root = (PROJECT_ROOT / p).resolve()
        resolved = from_root if is_within(from_root, base) else (base / p).resolve()
    if not is_within(resolved, base):
        raise ValueError(f"{raw!r} resolves outside {base}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(raw)
    return resolved
