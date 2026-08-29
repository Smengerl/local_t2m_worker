"""
CLI tool for cancelling a running (or stuck) job in the batch queue.

For a job owned by a *standalone* ``batch.worker`` process, this sends SIGTERM
(or SIGKILL with --force) to that process and marks the job failed.

For a job owned by the ``batch.server`` process (embedded worker), signalling
the PID would kill the whole web server.  This tool refuses to do that and
tells you to cancel via the dashboard or ``POST /api/jobs/{id}/cancel``
instead.  The two are told apart via ``batch/worker.pid``, which only a
standalone worker writes.

Usage:
    python -m batch.cancel <job-id>
    python -m batch.cancel <job-id> --force   # SIGKILL instead of SIGTERM
    python -m batch.cancel --list             # show running jobs and their PIDs
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch.queue import get_job, list_jobs, mark_failed
from batch.paths import WORKER_PID_FILE


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _standalone_worker_pid() -> int | None:
    """Return the PID of the running standalone worker, or None.

    None means: no standalone worker is running (the queue is being served by
    ``batch.server`` with an embedded worker, or nothing is running, or the
    pidfile is stale).
    """
    try:
        pid = int(WORKER_PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _cancel_job(job_id: str, force: bool = False) -> int:
    """Cancel a job by ID.  Returns an exit code (0 = success, 1 = error)."""
    job = get_job(job_id)
    if job is None:
        print(f"❌  Job {job_id!r} not found.", file=sys.stderr)
        return 1

    status = job.get("status")
    if status not in ("running", "pending"):
        print(f"ℹ️   Job {job_id[:8]}… is already {status!r} — nothing to cancel.")
        return 0

    if status == "pending":
        # Pending jobs have no worker yet — just mark them failed.
        mark_failed(job_id, "Cancelled by user before execution started.")
        print(f"✅  Pending job {job_id[:8]}… marked as cancelled.")
        return 0

    # ── Running job: try to terminate the worker process ─────────────────────
    pid = job.get("worker_pid")

    # If the recorded PID is already dead, the worker crashed — just clean up
    # the stale "running" entry, don't try to signal anything.
    if pid is not None and not _pid_alive(pid):
        mark_failed(job_id, f"Cancelled by user (worker PID {pid} was already gone).")
        print(f"✅  Worker PID {pid} is gone; job {job_id[:8]}… marked as cancelled.")
        return 0

    # Only signal a PID we can positively identify as a standalone worker.
    # Otherwise that PID is the web server (embedded worker) or something we
    # don't recognise — killing it would take the whole server down.
    standalone_pid = _standalone_worker_pid()
    if pid is not None and pid != standalone_pid:
        print(
            f"⚠️   Job {job_id[:8]}… is owned by PID {pid}, which is not the "
            "standalone worker.\n"
            "     It is most likely the web server's embedded worker — "
            "signalling it would kill the server.\n"
            "     Cancel it from the dashboard, or:\n"
            f"       curl -X POST http://<host>:<port>/api/jobs/{job['id']}/cancel\n"
            "     (If you are sure no server is running, the job is stale — "
            "delete it from the dashboard.)",
            file=sys.stderr,
        )
        return 1

    sig = signal.SIGKILL if force else signal.SIGTERM
    sig_name = "SIGKILL" if force else "SIGTERM"

    killed = False
    if pid is not None:
        try:
            os.kill(pid, sig)
            print(f"📨  Sent {sig_name} to worker process (PID {pid}).")
            killed = True

            # Wait up to 5 seconds for the process to exit.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)   # probe: raises if process is gone
                    time.sleep(0.2)
                except ProcessLookupError:
                    break
            else:
                if not force:
                    print(
                        f"⚠️   Worker PID {pid} did not exit within 5 s. "
                        "Re-run with --force to send SIGKILL.",
                        file=sys.stderr,
                    )

        except ProcessLookupError:
            print(f"⚠️   Worker PID {pid} no longer exists (already exited).")
        except PermissionError:
            print(
                f"❌  No permission to signal PID {pid}. "
                "Try running as the same user as the worker.",
                file=sys.stderr,
            )
    else:
        print("⚠️   No worker_pid stored for this job (worker may not have written it yet).")

    # ── Always mark the job as failed in the queue ────────────────────────────
    reason = (
        f"Cancelled by user (sent {sig_name} to PID {pid})."
        if killed and pid is not None
        else "Cancelled by user (process was already gone or PID unknown)."
    )
    mark_failed(job_id, reason)
    print(f"✅  Job {job_id[:8]}… marked as cancelled/failed in queue.")
    return 0


def _list_running() -> None:
    """Print all running jobs with their PIDs."""
    running = [j for j in list_jobs() if j.get("status") == "running"]
    if not running:
        print("No running jobs.")
        return
    print(f"{'ID':38}  {'PID':>7}  PROMPT")
    print("─" * 70)
    for j in running:
        pid_str = str(j.get("worker_pid") or "—").rjust(7)
        prompt = j.get("prompt", "")[:50]
        print(f"{j['id']}  {pid_str}  {prompt!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cancel a running or pending job in the batch queue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m batch.cancel abc123ef\n"
            "  python -m batch.cancel abc123ef --force\n"
            "  python -m batch.cancel --list\n"
        ),
    )
    parser.add_argument(
        "job_id",
        nargs="?",
        metavar="JOB_ID",
        help="ID (or prefix) of the job to cancel.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Send SIGKILL instead of SIGTERM (immediate, non-graceful).",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all currently running jobs and their worker PIDs, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        _list_running()
        return

    if not args.job_id:
        print("❌  Please provide a JOB_ID or use --list.", file=sys.stderr)
        sys.exit(1)

    # Support short prefixes: find first job whose ID starts with the given string.
    job_id = args.job_id
    if len(job_id) < 36:
        matches = [j for j in list_jobs() if j["id"].startswith(job_id)]
        if len(matches) == 0:
            print(f"❌  No job found with ID prefix {job_id!r}.", file=sys.stderr)
            sys.exit(1)
        if len(matches) > 1:
            print(
                f"❌  Ambiguous prefix {job_id!r} matches {len(matches)} jobs. "
                "Use a longer prefix.",
                file=sys.stderr,
            )
            sys.exit(1)
        job_id = matches[0]["id"]

    sys.exit(_cancel_job(job_id, force=args.force))


if __name__ == "__main__":
    main()
