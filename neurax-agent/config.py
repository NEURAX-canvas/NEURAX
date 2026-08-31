"""Shared configuration and state for neurax-agent."""
import asyncio
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RunEntry:
    task: "asyncio.Task[None]"
    queue: "asyncio.Queue[dict[str, Any]]"
    created_at: float


# Runtime state.
#
# `GET /runs/{id}/events` removes its own entry in a `finally` block once the
# stream ends or the client disconnects, and cancels the run's task there too
# — a client that is no longer listening has no more use for a run still in
# progress, and every further step is a real LLM call billed to whatever
# credentials the run started with. That's the normal path.
#
# But `_run_agent` runs as a fire-and-forget `asyncio.create_task` that never
# touches `_runs` itself, so a caller that never connects to `/events` at all
# has nothing to trigger that cleanup: the task keeps running to completion
# (bounded — MAX_ATTEMPTS retries and a per-call LLM timeout, not infinite —
# but still real, billed work for a request nobody is waiting on), and the
# queue it writes into stays in memory for the life of the process.
# `_sweep_expired_runs` (called periodically from app.py) is the fallback for
# exactly that case: it cancels the task and drops the entry.
_runs: dict[str, RunEntry] = {}
_RUN_RETENTION_SECONDS = 5 * 60

_catalogue_cache: dict[str, list[dict[str, Any]]] = {}
_CATALOGUE_CACHE_MAX = 32


def _sse_event(event: str, data: dict[str, Any]) -> str:
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stop_run(run_id: str) -> None:
    """Cancel a run's task (a no-op if it already finished) and drop its
    entry. Shared by the normal `/events` disconnect path and the sweeper so
    there is exactly one place that decides what "stopping a run" means.
    """
    entry = _runs.pop(run_id, None)
    if entry is not None and not entry.task.done():
        entry.task.cancel()


def _sweep_expired_runs(now: float | None = None) -> int:
    """Stop every `_runs` entry older than `_RUN_RETENTION_SECONDS`. Returns
    how many were reclaimed. Takes `now` explicitly so the retention boundary
    itself is directly testable, not just "a background task runs".
    """
    now = time.monotonic() if now is None else now
    expired = [
        run_id
        for run_id, entry in _runs.items()
        if now - entry.created_at > _RUN_RETENTION_SECONDS
    ]
    for run_id in expired:
        _stop_run(run_id)
    return len(expired)
