"""
Job-ready notification — shared event between the worker loop and the enqueue API.

``run_worker_async(keep_alive=True)`` calls ``init()`` once when it starts.
The jobs API router calls ``notify()`` after every enqueue so the worker wakes
up immediately.

Works identically whether the worker is started via the CLI
(``python -m batch.worker --keep-alive``) or embedded in the FastAPI server.
"""

import asyncio
import logging
from typing import Optional

log = logging.getLogger(__name__)

_event: Optional[asyncio.Event] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


def init() -> asyncio.Event:
    """Create (or return the existing) global job event.

    Idempotent — safe to call multiple times; always returns the same event
    for the lifetime of the current event loop.  Must be called from an async
    context so the running loop can be captured for thread-safe ``notify()``.
    """
    global _event, _loop
    if _event is None:
        _loop = asyncio.get_running_loop()
        _event = asyncio.Event()
    return _event


def get() -> Optional[asyncio.Event]:
    """Return the current event, or None if not yet initialised."""
    return _event


def notify() -> None:
    """Signal that a new pending job is available (thread-safe).

    No-op when the worker has not been started with ``keep_alive=True``
    (e.g. one-shot CLI mode).

    ``call_soon_threadsafe`` is correct from any calling context — both from
    FastAPI sync route handlers (which run in uvicorn's thread pool) and from
    async code on the event loop itself.  It is the standard way to wake an
    asyncio.Event from outside the loop.
    """
    if _event is None or _loop is None:
        return
    try:
        _loop.call_soon_threadsafe(_event.set)
    except RuntimeError:
        pass  # loop already closed during shutdown


def reset() -> None:
    """Clear module state — used in tests and between server restarts."""
    global _event, _loop
    _event = None
    _loop = None
