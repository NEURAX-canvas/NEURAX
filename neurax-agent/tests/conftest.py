"""Shared pytest fixtures for neurax-agent's test suite.

`agent_graph.run_agent_graph` calls `langchain_runner.plan_run_strategy`
(a real LLM call) unconditionally at the start of every run. Every test
that drives `run_agent_graph` without deliberately testing plan behavior
itself would otherwise make a real, slow, failing network attempt before
`plan_run_strategy`'s own try/except degrades it to `[]` — the same
"never make a real LLM call in a unit test" discipline every other
LLM-touching function in this codebase is already tested under
(`run_controller_step`, `plan_architecture`, `make_chat_model` itself),
just enforced once here instead of in every test file that happens to
exercise the loop.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def _no_real_plan_generation_by_default(monkeypatch):
    """Fast, deterministic "no roadmap" for every test by default — a real,
    already-handled state (`plan_run_strategy` itself returns `[]` on any
    failure), not a fake success. A test that specifically exercises plan
    behavior overrides this within its own body, matching how tests already
    override `agent_graph.run_controller_step` per test rather than relying
    on a shared default for that one.
    """
    try:
        import agent_graph

        async def _no_plan(**kwargs):
            return []

        monkeypatch.setattr(agent_graph, "plan_run_strategy", _no_plan)
    except ImportError:
        # A test file that never imports agent_graph (e.g. pure Rust-adjacent
        # or frontend-adjacent tooling, if any ever lands here) has nothing
        # to patch.
        pass
    yield
