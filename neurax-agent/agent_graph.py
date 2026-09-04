"""agent_graph.py — the step-by-step agentic loop.

`agent_runner.py`'s `_run_agent` plans a *whole* architecture in one
structured-output call, then materializes it deterministically. That is
correct and well-tested for "build me an X" — a single shot at a complete
design, bounded by `MAX_ATTEMPTS = 4` retries. It cannot do the other half of
what an autonomous agent needs: take one small step, look at the real result,
and decide the next one — the shape "plan and work for minutes at a time"
actually requires.

This module is that loop, built on LangGraph (`StateGraph`) rather than a
hand-rolled `while`, for the reasons recorded in the project's own plan
document: step-by-step execution, a real stop condition instead of an
unbounded loop, and streaming that maps directly onto the SSE events
`app.py`'s `/runs/{id}/events` already serves.

It reuses `langchain_runner.run_controller_step` — a genuine step-by-step
"here is the next tool call" function that predates this module and was
never called by anything (`grep` confirms zero callers before this file) —
as the model-calling half of the loop, rather than re-deriving its ~300
lines of prompt assembly. The one real bug fixed to make that reuse safe:
`run_controller_step` never accepted `credentials`, so calling it for real
would have silently billed the server's own key regardless of what the
caller supplied — the exact bring-your-own-key violation
`tests/test_credentials.py` exists to catch on `make_chat_model` itself.
Fixed at the source (`langchain_runner.py`), not worked around here.

Every tool call this loop plans is applied through the same
`snapshot_ops._apply_tool_to_snapshot` the 3-phase pipeline already uses —
a cycle, a fan-in violation, or an unknown node id is rejected there exactly
as it always was. This loop never trusts the model's stated intent over
what that function actually did: a rejection is read back
(`_last_tool_rejection`) and fed into the next step's history so the model
sees its own last action failed, instead of silently repeating it forever.

Safety ceilings — none of which existed anywhere in this codebase before
this module, Rust or Python — stop a run from running forever: `max_steps`
and `timeout_seconds`, both checked after every tool execution, both
overridable by the caller (`app.py`, once wired in) rather than hardcoded.
"""
from __future__ import annotations

import asyncio
import logging
import operator
import time
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

import analysis_tools
import memory_tools
import web_search_tools
from langchain_runner import ALL_TOOL_DESCRIPTIONS, run_controller_step
from snapshot_ops import _apply_tool_to_snapshot

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [NEURAX-GRAPH] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


#: The old 3-phase pipeline was bounded only implicitly (MAX_ATTEMPTS=4
#: planning retries, each behind a per-call LLM timeout). A step-by-step loop
#: has no such natural ceiling of its own — these are it.
DEFAULT_MAX_STEPS = 60
DEFAULT_TIMEOUT_SECONDS = 300.0

#: A full hyperparameter sweep is many /analyze-equivalent compiler runs,
#: not one — capped separately from, and more tightly than, the overall
#: step ceiling, so a loop can't spend its whole step budget re-sweeping
#: the same design over and over.
DEFAULT_MAX_EXPENSIVE_CALLS = 3
_EXPENSIVE_ANALYSIS_TOOLS = frozenset({"find_optimal_hyperparameters"})

DEFAULT_MODE = "creation"

#: Least-privilege tool grants, one set per mode — the structural half of
#: mode enforcement. `execute_tool` checks a planned tool's name against
#: this set (never a canvas-mutation tool for `explanation`, never
#: `add_node`/`delete_node`/`connect` for `optimization`) *before* it ever
#: reaches `_apply_tool_to_snapshot` or `analysis_tools.dispatch` — a
#: hallucinated call to an ungranted tool is rejected here even though
#: `langchain_runner._build_tools_section` already keeps the model from
#: being told about it in the first place. Real principle, not
#: hypothetical: OpenClaw's own permission model works the same way
#: ("tools simply don't exist in restricted modes, not merely discouraged
#: in prose") — see this project's plan document's research notes.
#:
#: `"done"` is not listed in any set — every mode needs a way to stop,
#: checked as a special case in `execute_tool` rather than repeated in each.
MODE_TOOL_GRANTS: dict[str, frozenset[str]] = {
    "creation": frozenset({
        "set_family", "add_node", "connect", "disconnect", "delete_node",
        "set_node_params", "set_hw_config", "initialize_hyperparams", "set_hyperparams",
        "navigate_to", "run_analysis", "select_node",
        "analyze_architecture", "check_budget",
        "remember_preference",
    }),
    "optimization": frozenset({
        # Deliberately no add_node/delete_node/connect/disconnect/set_family:
        # this mode tunes an existing design, it does not redesign it.
        "set_node_params", "set_hw_config", "set_hyperparams",
        "navigate_to", "run_analysis", "select_node",
        "analyze_architecture", "check_budget", "find_optimal_hyperparameters",
        "estimate_training_cost", "get_hardware_list",
        "remember_preference",
    }),
    "research": frozenset({
        "set_family", "add_node", "connect", "disconnect", "delete_node",
        "set_node_params", "set_hw_config", "initialize_hyperparams", "set_hyperparams",
        "navigate_to", "run_analysis", "select_node",
        "analyze_architecture", "check_budget", "find_optimal_hyperparameters",
        "get_presets", "get_preset", "get_hardware_list", "estimate_training_cost",
        "remember_preference", "search_past_designs",
        # "web_search" is NOT listed here — it's added dynamically in
        # run_agent_graph, only when the caller actually supplied a Tavily
        # key. Listing it unconditionally would grant a tool this run can't
        # actually use, the opposite of what MODE_TOOL_GRANTS is for.
    }),
    "explanation": frozenset({
        # Read-only: no canvas-mutation tool anywhere in this set —
        # search_past_designs included (it only reads), remember_preference
        # deliberately not (it writes).
        "navigate_to", "select_node",
        "analyze_architecture", "get_compliance_config", "get_presets", "get_preset",
        "explain_layer_type", "search_past_designs",
    }),
}

#: Every name in every grant above must be a real, described tool — a typo
#: here would silently gate on a tool that can never be granted. Checked at
#: import time, not buried in a test someone has to remember to run.
_UNKNOWN_GRANTED_TOOLS = {
    name
    for grant in MODE_TOOL_GRANTS.values()
    for name in grant
    if name not in ALL_TOOL_DESCRIPTIONS
}
assert not _UNKNOWN_GRANTED_TOOLS, f"MODE_TOOL_GRANTS names undescribed tools: {_UNKNOWN_GRANTED_TOOLS}"


class AgentGraphState(TypedDict):
    run_id: str
    user_message: str
    snapshot: dict[str, Any]
    history: list[dict[str, Any]]
    credentials: Optional[dict[str, Any]]
    #: Resolved once from `credentials.get("search_api_key")` in
    #: `run_agent_graph` — kept on the state rather than re-read from
    #: `credentials` on every `web_search` call, the same reasoning as
    #: `allowed_tools` below (one resolution, agreed on for the whole run).
    search_api_key: Optional[str]
    #: Real, only ever a value when the canvas is a saved project — see
    #: `memory_tools.py`'s and `agent_memory.rs`'s module docs for why
    #: memory is scoped by this alone, not a user id. `None` means "this
    #: canvas has no project to remember against," a normal state, not
    #: an error — every memory tool degrades gracefully when it's `None`.
    project_id: Optional[str]
    #: Fetched once at the very start of the run (`run_agent_graph`) via
    #: `memory_tools.get_core_preferences` — always shown in the prompt,
    #: never re-fetched mid-run.
    core_memory: list[str]
    mode: str
    #: Resolved once in `run_agent_graph` from `MODE_TOOL_GRANTS[mode]` —
    #: kept on the state (not re-looked-up per step) so `plan_step` and
    #: `execute_tool` always agree on exactly the same set within one run,
    #: even if `MODE_TOOL_GRANTS` were hot-reloaded mid-process.
    allowed_tools: frozenset[str]
    step_count: int
    max_steps: int
    max_expensive_calls: int
    #: How many times each analysis tool has actually been called this run
    #: (not attempted — a call blocked by `max_expensive_calls` doesn't
    #: increment its own ceiling further). Only `execute_tool` writes this,
    #: so a plain "last write wins" field is enough — no reducer needed.
    tool_call_counts: dict[str, int]
    started_at: float
    timeout_seconds: float
    last_tool: Optional[dict[str, Any]]
    stop_reason: Optional[str]
    #: Every node appends the SSE-shaped events *it* produced this step;
    #: `operator.add` concatenates rather than the default "last write wins"
    #: reducer, so `run_agent_graph`'s `astream(..., )` (default "updates"
    #: mode: `{node_name: partial_update}` per step, not the whole
    #: accumulated state) sees exactly the new events each step made, once.
    events: Annotated[list[dict[str, Any]], operator.add]


def _event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": event_type, "data": data}


async def plan_step(state: AgentGraphState) -> dict[str, Any]:
    """Ask the model for exactly one next tool call, from exactly this
    mode's granted vocabulary — `allowed_tools` reaches `run_controller_step`
    here so the model is never even told about a tool it isn't allowed to
    use; `execute_tool` enforces the same set again regardless."""
    result = await run_controller_step(
        user_message=state["user_message"],
        snapshot=state["snapshot"],
        history=state["history"],
        credentials=state.get("credentials"),
        allowed_tools=state["allowed_tools"],
        mode=state["mode"],
        core_memory=state.get("core_memory"),
    )
    assistant_text = str(result.get("assistant") or "")
    tool = result.get("tool") or {"name": "done", "args": {}}

    events: list[dict[str, Any]] = []
    if assistant_text:
        events.append(_event("assistant", {"content": assistant_text}))

    return {"last_tool": tool, "events": events}


def _describe_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


async def _execute_analysis_tool(state: AgentGraphState, tool: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """The analysis half of `execute_tool` — never touches the snapshot;
    its result is read back into history as a `system` turn, the same
    channel a canvas-tool rejection already uses to tell the next
    `plan_step` call what actually happened."""
    name = str(tool.get("name") or "")
    args = tool.get("args") or {}
    counts = dict(state.get("tool_call_counts") or {})

    if name in _EXPENSIVE_ANALYSIS_TOOLS and counts.get(name, 0) >= state["max_expensive_calls"]:
        result_text = (
            f"'{name}' has already run {counts.get(name, 0)} time(s) this run — that's "
            f"this run's limit ({state['max_expensive_calls']}). Work with what you "
            "already know, or call `done`."
        )
        logger.warning(f"⛔ STEP {state['step_count'] + 1}: '{name}' blocked — per-run ceiling reached")
    else:
        try:
            result_text = await analysis_tools.dispatch(name, args, state["snapshot"])
            logger.info(f"📊 STEP {state['step_count'] + 1}: {name}({_describe_args(args)}) -> {len(result_text)} chars")
        except Exception as e:
            result_text = f"'{name}' failed: {e}"
            logger.error(f"💥 STEP {state['step_count'] + 1}: '{name}' raised: {e}")
        if name in _EXPENSIVE_ANALYSIS_TOOLS:
            counts[name] = counts.get(name, 0) + 1

    history.append({"role": "system", "content": f"Result of {name}: {result_text}"})

    return {
        "snapshot": state["snapshot"],
        "history": history,
        "step_count": state["step_count"] + 1,
        "tool_call_counts": counts,
        "events": [_event("tool", tool), _event("tool_result", {"tool": name, "content": result_text})],
    }


def _tool_result_update(
    state: AgentGraphState, tool: dict[str, Any], history: list[dict[str, Any]], result_text: str
) -> dict[str, Any]:
    """The shared shape every non-canvas, non-`_execute_analysis_tool` tool
    branch returns: the result read back into history for the next
    `plan_step` call, and both the `tool` and `tool_result` SSE events —
    factored out once this pattern reached its fourth near-identical copy
    (`explain_layer_type`, `web_search`, `remember_preference`,
    `search_past_designs`)."""
    name = str(tool.get("name") or "")
    history.append({"role": "system", "content": f"Result of {name}: {result_text}"})
    return {
        "snapshot": state["snapshot"],
        "history": history,
        "step_count": state["step_count"] + 1,
        "events": [_event("tool", tool), _event("tool_result", {"tool": name, "content": result_text})],
    }


def _explain_layer_type(snapshot: dict[str, Any], layer_type: str) -> str:
    """Look up NEURAX's own description of a block type, sourced from the
    snapshot's own `catalogue[].description` — which
    `neurax-ui/src/pages/Index.tsx::agentGetSnapshot` populates straight
    from `registry.ts`'s real per-block descriptions (560 entries). No
    second, backend-side copy of that text exists; if the snapshot carries
    none (an older frontend build, or a caller that never sent one), that is
    reported plainly rather than guessed at."""
    for item in snapshot.get("catalogue") or []:
        if isinstance(item, dict) and item.get("type") == layer_type:
            desc = item.get("description")
            if desc:
                return f"{layer_type}: {desc}"
            return f"'{layer_type}' is in the catalogue but has no description recorded."
    return f"'{layer_type}' was not found in the current catalogue."


async def execute_tool(state: AgentGraphState) -> dict[str, Any]:
    """Apply the planned tool — to the canvas if it's a mutation, to the
    compiler if it's an analysis call (`analysis_tools.ANALYSIS_TOOL_NAMES`).
    `_apply_tool_to_snapshot` is the same validation `agent_runner.py`'s
    materialization already goes through for canvas edits: a cycle or a
    fan-in violation is rejected there exactly as it always was, regardless
    of what the model intended.

    The very first check, before any of that, is the structural
    least-privilege gate: a tool outside `state["allowed_tools"]` (`"done"`
    always excepted) is refused here even if `plan_step` somehow still
    named it — the prompt-level filtering in `_build_tools_section` is a
    courtesy to the model, this is the actual boundary."""
    tool = state.get("last_tool") or {"name": "done", "args": {}}
    name = str(tool.get("name") or "done")
    args = tool.get("args") or {}

    history = list(state["history"])
    history.append({"role": "assistant", "content": f"Called {name}({_describe_args(args)})"})

    if name != "done" and name not in state["allowed_tools"]:
        message = f"'{name}' is not available in '{state['mode']}' mode."
        history.append({"role": "system", "content": message})
        logger.warning(f"⛔ STEP {state['step_count'] + 1}: '{name}' refused — outside '{state['mode']}' mode's grant")
        return {
            "snapshot": state["snapshot"],
            "history": history,
            "step_count": state["step_count"] + 1,
            "events": [_event("tool_result", {"tool": name, "content": message})],
        }

    if name == "web_search":
        # Reaching here at all already proves a key was present: 'web_search'
        # is only ever in state["allowed_tools"] when run_agent_graph found
        # one (see there) — this is not itself the credential check.
        result_text = await web_search_tools.web_search(
            query=str(args.get("query") or ""),
            api_key=state.get("search_api_key") or "",
        )
        return _tool_result_update(state, tool, history, result_text)

    if name == "explain_layer_type":
        result_text = _explain_layer_type(state["snapshot"], str(args.get("layer_type") or ""))
        return _tool_result_update(state, tool, history, result_text)

    if name == "remember_preference":
        preference = str(args.get("preference") or "").strip()
        project_id = state.get("project_id")
        if not project_id:
            result_text = "Nothing to remember against — this canvas isn't saved as a project yet."
        elif not preference:
            result_text = "No preference text given."
        else:
            saved = await memory_tools.add_core_preference(project_id, preference)
            result_text = f"Remembered: {preference}" if saved else "Could not save this — memory is unavailable right now."
        return _tool_result_update(state, tool, history, result_text)

    if name == "search_past_designs":
        project_id = state.get("project_id")
        if not project_id:
            result_text = "This canvas isn't saved as a project yet — no history to search."
        else:
            entries = await memory_tools.search_past_designs(project_id, str(args.get("query") or ""))
            if entries is None:
                # Distinct from "genuinely no matches" — memory_tools.py's
                # own docstring on why this one function doesn't collapse
                # failure into an empty list the way the others do.
                result_text = "Could not search past designs — memory is unavailable right now."
            elif entries:
                result_text = "Past designs found:\n" + "\n".join(f"- {e}" for e in entries)
            else:
                result_text = "No past designs found for this project."
        return _tool_result_update(state, tool, history, result_text)

    if name in analysis_tools.ANALYSIS_TOOL_NAMES:
        return await _execute_analysis_tool(state, tool, history)

    # `_apply_tool_to_snapshot` mutates its argument as well as returning it
    # — passing a copy keeps this node's input/output relationship explicit
    # rather than relying on the caller not noticing the aliasing.
    snapshot = _apply_tool_to_snapshot(dict(state["snapshot"]), tool)

    rejection = snapshot.pop("_last_tool_rejection", None)
    if rejection:
        history.append({
            "role": "system",
            "content": f"'{name}' was rejected: {rejection.get('message', 'unknown reason')}",
        })
        logger.warning(f"⚠️ STEP {state['step_count'] + 1}: '{name}' rejected — {rejection.get('reason')}")
    else:
        logger.info(f"🔧 STEP {state['step_count'] + 1}: {name}({_describe_args(args)})")

    return {
        "snapshot": snapshot,
        "history": history,
        "step_count": state["step_count"] + 1,
        "events": [_event("tool", tool)],
    }


def should_continue(state: AgentGraphState) -> str:
    """Routing only — no state update. `finish` re-derives *why* it was
    reached from the same three checks, since a conditional edge's return
    value can only select the next node, not carry data forward."""
    last_tool = state.get("last_tool") or {}
    if str(last_tool.get("name") or "") == "done":
        return "finish"
    if state["step_count"] >= state["max_steps"]:
        return "finish"
    if time.monotonic() - state["started_at"] >= state["timeout_seconds"]:
        return "finish"
    return "plan_step"


async def finish(state: AgentGraphState) -> dict[str, Any]:
    last_tool = state.get("last_tool") or {}
    elapsed = time.monotonic() - state["started_at"]

    if str(last_tool.get("name") or "") == "done":
        reason = "done"
        message: Optional[str] = None  # the model's own last narration already said it
    elif state["step_count"] >= state["max_steps"]:
        reason = "step_limit"
        message = (
            f"Stopped after {state['step_count']} steps, this run's limit, without "
            "calling `done`. What's built so far is on the canvas — ask me to "
            "continue if it needs more work."
        )
    else:
        reason = "timeout"
        message = (
            f"Stopped after {elapsed:.0f}s, this run's time limit, without calling "
            "`done`. What's built so far is on the canvas — ask me to continue if "
            "it needs more work."
        )

    logger.info(f"🏁 GRAPH RUN FINISHED | reason={reason} | steps={state['step_count']} | elapsed={elapsed:.1f}s")
    events = [_event("assistant", {"content": message})] if message else []
    return {"stop_reason": reason, "events": events}


_compiled_graph = None


def build_graph():
    graph = StateGraph(AgentGraphState)
    graph.add_node("plan_step", plan_step)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("finish", finish)

    graph.add_edge(START, "plan_step")
    graph.add_edge("plan_step", "execute_tool")
    graph.add_conditional_edges(
        "execute_tool",
        should_continue,
        {"plan_step": "plan_step", "finish": "finish"},
    )
    graph.add_edge("finish", END)

    return graph.compile()


def get_graph():
    """Built once per process — a `StateGraph` compiles into an immutable
    execution plan, and every run gets its own fresh state dict, so nothing
    about the compiled graph itself is per-run."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_agent_graph(
    run_id: str,
    q: "asyncio.Queue[dict[str, Any]]",
    user_message: str,
    snapshot: dict[str, Any],
    credentials: Optional[dict[str, Any]] = None,
    conversation_history: Optional[list[dict[str, Any]]] = None,
    mode: str = DEFAULT_MODE,
    project_id: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_expensive_calls: int = DEFAULT_MAX_EXPENSIVE_CALLS,
) -> None:
    """Drive the graph for one run, forwarding every event it produces to
    the same SSE queue `agent_runner.py::_run_agent` already writes to.
    `app.py`'s `/runs/{id}/events` does not know or care which orchestrator
    produced an event — only that the stream ends in a `done` event, which
    this function guarantees exactly like `_run_agent` does, `finally`.

    `mode` resolves to a tool grant via `MODE_TOOL_GRANTS`; an unrecognized
    mode falls back to `DEFAULT_MODE` rather than raising — a stale or
    future frontend build naming a mode this process doesn't know yet
    degrades to the safe default instead of failing the run outright.

    `project_id`, when given, is what makes this run's memory real: core
    preferences and recent conversation are recalled before the loop starts,
    and this run's own turns (plus, on a successful creation/research run, a
    short archival summary) are saved after it ends — all best-effort, all
    inside the same `finally` that guarantees `done` fires, so a memory
    failure never blocks the run's own result from reaching the caller.
    """
    allowed_tools = MODE_TOOL_GRANTS.get(mode, MODE_TOOL_GRANTS[DEFAULT_MODE])

    # web_search is granted dynamically, not listed in MODE_TOOL_GRANTS
    # itself: only research mode, and only when the caller actually
    # supplied their own Tavily key — confirmed BYOK, no server-side
    # fallback. A mode/credentials combination that doesn't meet both
    # simply never sees this tool, no different from any other ungranted
    # one — see MODE_TOOL_GRANTS["research"]'s own comment.
    search_api_key = (credentials or {}).get("search_api_key")
    if mode == "research" and search_api_key:
        allowed_tools = allowed_tools | {"web_search"}

    logger.info(
        f"🚀 GRAPH RUN STARTED | run_id={run_id} | mode={mode} | project_id={project_id} | "
        f"max_steps={max_steps} | timeout={timeout_seconds}s"
    )

    # Everything from here on — recall, the graph run itself, and the
    # persistence in `finally` below — lives inside one try/finally so
    # `done` is guaranteed to reach the caller (below) no matter where a
    # cancellation or exception lands, recall's own network calls included.
    # An earlier version of this function fetched recall *before* this
    # block: a client disconnecting while that fetch was in flight would
    # have propagated `CancelledError` straight out of this function with
    # `done` never queued at all — the exact invariant every test in this
    # module checks for, just from an angle none of them were reaching.
    core_memory: list[str] = []
    recalled_turns: list[dict[str, str]] = []
    # Tracked as the stream goes by so the `finally` block below can persist
    # recall/archival memory from whatever the run actually produced —
    # `history`/`stop_reason` are "last write wins" fields (no reducer), so
    # the most recent update carrying either is the final value; `assistant`
    # events are narration text, collected here rather than re-derived from
    # `history` (which only ever holds mechanical "Called X(...)" entries,
    # never the model's own words — see `plan_step`/`execute_tool`).
    final_history: list[dict[str, Any]] = list(conversation_history or [])
    final_stop_reason = "unknown"
    assistant_narrations: list[str] = []

    try:
        if project_id:
            core_memory = await memory_tools.get_core_preferences(project_id)
            recalled_turns = await memory_tools.get_recent_conversation(project_id)

        initial_state: AgentGraphState = {
            "run_id": run_id,
            "user_message": user_message,
            "snapshot": snapshot,
            "history": recalled_turns + list(conversation_history or []),
            "credentials": credentials,
            "search_api_key": search_api_key,
            "project_id": project_id,
            "core_memory": core_memory,
            "mode": mode,
            "allowed_tools": allowed_tools,
            "step_count": 0,
            "max_steps": max_steps,
            "max_expensive_calls": max_expensive_calls,
            "tool_call_counts": {},
            "started_at": time.monotonic(),
            "timeout_seconds": timeout_seconds,
            "last_tool": None,
            "stop_reason": None,
            "events": [],
        }
        final_history = list(initial_state["history"])

        graph = get_graph()
        async for step in graph.astream(initial_state):
            for _node_name, update in step.items():
                if not update:
                    continue
                if "history" in update:
                    final_history = update["history"]
                if update.get("stop_reason"):
                    final_stop_reason = update["stop_reason"]
                for evt in update.get("events") or []:
                    await q.put(evt)
                    if evt.get("event") == "assistant":
                        content = (evt.get("data") or {}).get("content")
                        if content:
                            assistant_narrations.append(str(content))
    except asyncio.CancelledError:
        # A caller disconnecting (`config._stop_run`) cancels this task —
        # that is normal shutdown, not a failure to report as one.
        raise
    except Exception as e:
        logger.error(f"💥 GRAPH RUN FAILED: {e}", exc_info=True)
        await q.put(_event("error", {"message": str(e)}))
        await q.put(_event("assistant", {"content": f"I hit an error while working on this: {e}"}))
    finally:
        if project_id:
            try:
                turns = [{"role": "user", "content": user_message}]
                if assistant_narrations:
                    turns.append({"role": "assistant", "content": assistant_narrations[-1]})
                await memory_tools.append_conversation_turns(project_id, turns)

                if final_stop_reason == "done" and mode in ("creation", "research") and assistant_narrations:
                    summary = " ".join(assistant_narrations)[:2000]
                    await memory_tools.add_archival_entry(project_id, summary)
            except Exception as mem_err:
                # Memory is a real feature, not a dependency the run's own
                # result relies on — a failure here must never keep `done`
                # from reaching the caller below.
                logger.warning(f"post-run memory persistence failed: {mem_err}")
        await q.put(_event("done", {}))
