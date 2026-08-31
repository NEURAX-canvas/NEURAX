"""A run whose caller never connects to /events, or disconnects mid-run, must
actually stop — not just have its bookkeeping entry dropped.

Before this, `_run_agent` ran as a fire-and-forget `asyncio.create_task` that
`_runs` never referenced back to: nothing could cancel it, and nothing ever
removed its entry unless the caller happened to drain `/events` to "done".
An abandoned run kept making real, billed LLM calls to completion, and its
queue stayed in memory for the life of the process either way.
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


async def _forever():
    await asyncio.sleep(3600)


def _make_entry(created_at: float) -> config.RunEntry:
    task = asyncio.get_event_loop().create_task(_forever())
    return config.RunEntry(task=task, queue=asyncio.Queue(), created_at=created_at)


def test_sweep_cancels_and_drops_only_expired_runs():
    async def run():
        config._runs.clear()
        now = time.monotonic()

        old_id, recent_id = "old-run", "recent-run"
        config._runs[old_id] = _make_entry(now - config._RUN_RETENTION_SECONDS - 1)
        config._runs[recent_id] = _make_entry(now - 1)

        reclaimed = config._sweep_expired_runs(now=now)
        # Let the cancellation actually land before checking task state.
        await asyncio.sleep(0)

        assert reclaimed == 1
        assert old_id not in config._runs, "expired run's entry should be dropped"
        assert recent_id in config._runs, "recent run should survive the sweep"
        assert config._runs[recent_id].task.cancelled() is False

        # Cleanup: don't leak the still-running task past this test.
        config._stop_run(recent_id)
        await asyncio.sleep(0)

    asyncio.run(run())


def test_an_entry_exactly_at_the_boundary_survives():
    async def run():
        config._runs.clear()
        now = time.monotonic()
        entry_id = "boundary-run"
        config._runs[entry_id] = _make_entry(now - config._RUN_RETENTION_SECONDS)

        reclaimed = config._sweep_expired_runs(now=now)

        assert reclaimed == 0
        assert entry_id in config._runs

        config._stop_run(entry_id)
        await asyncio.sleep(0)

    asyncio.run(run())


def test_stop_run_cancels_the_task_and_removes_the_entry():
    async def run():
        config._runs.clear()
        entry_id = "abandoned-run"
        config._runs[entry_id] = _make_entry(time.monotonic())
        task = config._runs[entry_id].task

        config._stop_run(entry_id)
        await asyncio.sleep(0)  # give the cancellation a turn to be delivered

        assert entry_id not in config._runs
        assert task.cancelled() or task.done()

    asyncio.run(run())


def test_stop_run_on_an_already_finished_task_is_a_harmless_no_op():
    async def run():
        config._runs.clear()
        entry_id = "finished-run"

        async def _quick():
            return None

        task = asyncio.get_event_loop().create_task(_quick())
        await task  # ensure it's really done before stopping it
        config._runs[entry_id] = config.RunEntry(
            task=task, queue=asyncio.Queue(), created_at=time.monotonic()
        )

        config._stop_run(entry_id)  # must not raise

        assert entry_id not in config._runs

    asyncio.run(run())
