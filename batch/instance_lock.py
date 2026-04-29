"""
Exclusive process-level lock for the image-generation worker/server.

Both ``batch.worker`` (CLI) and ``batch.server`` (FastAPI) acquire this lock
at startup.  Because the lock is exclusive and non-blocking (timeout=0), the
second process to start will immediately exit with a clear error message
instead of running a second worker and causing double-generation races.

The lock file is ``<project_root>/batch.lock``.  It is automatically released
by the OS when the holding process exits — no explicit cleanup required.

Usage::

    from batch.instance_lock import acquire_exclusive

    _lock = acquire_exclusive("server")   # exits on conflict
    # ... start server / worker ...
    # _lock must stay in scope for the lifetime of the process
"""

import sys
from pathlib import Path

from filelock import FileLock, Timeout

_LOCK_FILE = Path(__file__).parent.parent / "batch.lock"


def acquire_exclusive(role: str) -> FileLock:
    """Try to acquire the exclusive instance lock.

    Args:
        role: Human-readable name of the caller ("server" or "worker").
              Used in the error message when the lock is already held.

    Returns:
        The acquired ``FileLock`` object.  The caller **must** keep a
        reference to it for the entire lifetime of the process — letting it
        be garbage-collected releases the lock.

    Raises:
        SystemExit(1): if another process already holds the lock.
    """
    lock = FileLock(str(_LOCK_FILE), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        other = "server" if role == "worker" else "worker"
        print(
            f"\n[{role}] ✖  Cannot start — another instance (batch.{other} or batch.{role}) "
            f"is already running.\n"
            f"         Stop the existing process first, then retry.\n"
            f"         Lock file: {_LOCK_FILE}\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return lock
