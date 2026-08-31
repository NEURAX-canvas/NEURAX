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
from typing import Any

from dotenv import load_dotenv
load_dotenv()  # Load .env before other imports

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import RunEntry, _runs, _sse_event, _stop_run, _sweep_expired_runs
from agent_runner import _run_agent


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


class ConversationTurn(BaseModel):
    #: "user" or "assistant" — mirrors the SSE `assistant` events this same
    #: agent already emits, which the frontend renders as chat and is
    #: therefore the natural source for this on the next request.
    role: str
    content: str


class RunRequest(BaseModel):
    user_message: str
    snapshot: CanvasSnapshot
    creativity: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Prior turns in this conversation, oldest first. Without this, a
    #: follow-up like "actually make it use GQA instead" or "you said this
    #: might not fit — try int8" has nothing to resolve "actually"/"you
    #: said" against beyond the current canvas structure, which captures
    #: what was built, not what was discussed or why.
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    #: Omit to fall back to the server's own credentials, where configured.
    credentials: LlmCredentials | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs")
async def create_run(req: RunRequest) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    snapshot = req.snapshot.model_dump()
    history = [turn.model_dump() for turn in req.conversation_history]
    task = asyncio.create_task(
        _run_agent(
            run_id,
            q,
            req.user_message,
            snapshot,
            creativity=req.creativity,
            credentials=req.credentials.model_dump() if req.credentials else None,
            conversation_history=history,
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
