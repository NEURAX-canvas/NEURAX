"""Neurax Agent - FastAPI entry point.

Code is split into modules:
- graph_utils.py: Graph topology analysis
- suggestions.py: Action suggestion functions
- prompts.py: LLM prompt builders
- snapshot_ops.py: Snapshot manipulation
- agent_runner.py: Main agent orchestration
- config.py: Shared state and utilities
"""
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
load_dotenv()  # Load .env before other imports

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import (
    RunEntry,
    _runs,
    _sse_event,
    _stop_run,
    _sweep_expired_runs,
    check_rate_limit,
    _RATE_LIMIT_MAX_RUNS,
    _RATE_LIMIT_WINDOW_SECONDS,
)
from agent_graph import (
    DEFAULT_MAX_EXPENSIVE_CALLS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODE,
    DEFAULT_TIMEOUT_SECONDS,
    run_agent_graph,
)

#: Read once at import time, matching how every other env-configurable
#: constant in this file/config.py already works. AGENT_MAX_STEPS existed
#: in deployments' .env files already (this repo's own local one included)
#: with nothing reading it — this run loop's step ceiling was always the
#: hardcoded default regardless of what an operator set it to.
_AGENT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", str(DEFAULT_MAX_STEPS)))
_AGENT_TIMEOUT_SECONDS = float(os.environ.get("AGENT_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
_AGENT_MAX_EXPENSIVE_CALLS = int(os.environ.get("AGENT_MAX_EXPENSIVE_CALLS", str(DEFAULT_MAX_EXPENSIVE_CALLS)))


async def _sweep_loop() -> None:
    while True:
        await asyncio.sleep(60)
        _sweep_expired_runs()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    sweeper = asyncio.create_task(_sweep_loop())
    try:
        yield
    finally:
        sweeper.cancel()


app = FastAPI(title="neurax-agent", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CanvasSnapshot(BaseModel):
    family: str = "transformer"
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    connections: list[dict[str, Any]] = Field(default_factory=list)
    groups: list[dict[str, Any]] = Field(default_factory=list)
    allowed_layer_types: list[str] = Field(default_factory=list)
    allowed_families: list[str] = Field(default_factory=list)
    catalogue_id: str | None = None
    catalogue: list[dict[str, Any]] = Field(default_factory=list)
    missing_mandatory_fields: list[str] = Field(default_factory=list)
    hw_config: dict[str, Any] = Field(default_factory=dict)
    analysis_warnings: list[dict[str, Any]] = Field(default_factory=list)


class LlmCredentials(BaseModel):
    """The caller's own model credentials.

    NEURAX is bring-your-own-key: the studio asks each user for a key, and until
    now never sent it, so every run was billed to whatever key the server had in
    its environment. Passing them per request makes the stated model true and
    keeps a public deployment from funding everyone's inference.
    """

    api_key: str
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    #: The caller's own Tavily key for research mode's `web_search` tool —
    #: same BYOK philosophy as `api_key` above, confirmed with the user as
    #: user-paid rather than server-side. Omit and research mode simply
    #: doesn't get the tool bound (`agent_graph.run_agent_graph`), rather
    #: than the run failing.
    search_api_key: str | None = None


class ConversationTurn(BaseModel):
    #: "user" or "assistant" — mirrors the SSE `assistant` events this same
    #: agent already emits, which the frontend renders as chat and is
    #: therefore the natural source for this on the next request.
    role: str
    content: str


class RunRequest(BaseModel):
    user_message: str
    snapshot: CanvasSnapshot
    #: Replaces the old `creativity` float entirely (retired, not kept as a
    #: secondary knob — each of these is a real, separate least-privilege
    #: tool grant on the backend, see `agent_graph.MODE_TOOL_GRANTS`, not a
    #: cosmetic label). Defaults to "creation" so an older frontend build
    #: that never sends this field still gets sensible behavior.
    mode: Literal["creation", "optimization", "research", "explanation"] = DEFAULT_MODE
    #: Prior turns in this conversation, oldest first. Without this, a
    #: follow-up like "actually make it use GQA instead" or "you said this
    #: might not fit — try int8" has nothing to resolve "actually"/"you
    #: said" against beyond the current canvas structure, which captures
    #: what was built, not what was discussed or why.
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    #: Omit to fall back to the server's own credentials, where configured.
    credentials: LlmCredentials | None = None
    #: The id of the currently-open, saved project (`Index.tsx`'s
    #: `currentProjectId`) — the key every memory tier is scoped by (see
    #: `agent_graph.py`'s `AgentGraphState.project_id` and `agent_memory.rs`'s
    #: module doc for why this and not a user id). `None` for a canvas that
    #: was never saved as a project — memory is then simply unavailable for
    #: this run, not an error.
    project_id: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs")
async def create_run(req: RunRequest, request: Request) -> dict[str, Any]:
    # The caller's own API key (when given) identifies them across IPs and
    # proxies more reliably than the socket address; falling back to the
    # socket address means an unauthenticated deployment still gets *some*
    # limiting rather than none.
    client_id = (
        (req.credentials.api_key if req.credentials else None)
        or (request.client.host if request.client else "unknown")
    )
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: at most {_RATE_LIMIT_MAX_RUNS} runs per "
                f"{_RATE_LIMIT_WINDOW_SECONDS:.0f}s. Wait and try again."
            ),
        )

    run_id = str(uuid.uuid4())
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    snapshot = req.snapshot.model_dump()
    history = [turn.model_dump() for turn in req.conversation_history]
    task = asyncio.create_task(
        run_agent_graph(
            run_id,
            q,
            req.user_message,
            snapshot,
            credentials=req.credentials.model_dump() if req.credentials else None,
            conversation_history=history,
            mode=req.mode,
            project_id=req.project_id,
            max_steps=_AGENT_MAX_STEPS,
            timeout_seconds=_AGENT_TIMEOUT_SECONDS,
            max_expensive_calls=_AGENT_MAX_EXPENSIVE_CALLS,
        )
    )
    _runs[run_id] = RunEntry(task=task, queue=q, created_at=time.monotonic())
    return {"run_id": run_id}


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    entry = _runs.get(run_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    q = entry.queue

    async def gen():
        try:
            while True:
                item = await q.get()
                event = str(item.get("event") or "message")
                data = item.get("data")
                if not isinstance(data, dict):
                    data = {"value": data}
                yield _sse_event(event, data)
                if event == "done":
                    break
        finally:
            # Covers both a normal finish (the task is already done, so
            # cancel() is a harmless no-op) and a client that disconnected
            # mid-run — in that case, there's no one left to spend further
            # LLM calls for, so the run stops right here instead of running
            # to completion unwatched.
            _stop_run(run_id)

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AGENT_PORT", "8099"))
    host = os.environ.get("AGENT_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="info")
