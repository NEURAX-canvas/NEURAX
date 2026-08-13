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
import uuid
from typing import Any

from dotenv import load_dotenv
load_dotenv()  # Load .env before other imports

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import _runs, _sse_event
from agent_runner import _run_agent

app = FastAPI(title="neurax-agent")

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


class RunRequest(BaseModel):
    user_message: str
    snapshot: CanvasSnapshot
    creativity: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Omit to fall back to the server's own credentials, where configured.
    credentials: LlmCredentials | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs")
async def create_run(req: RunRequest) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _runs[run_id] = q

    snapshot = req.snapshot.model_dump()
    asyncio.create_task(
        _run_agent(
            run_id,
            q,
            req.user_message,
            snapshot,
            _runs,
            creativity=req.creativity,
            credentials=req.credentials.model_dump() if req.credentials else None,
        )
    )
    return {"run_id": run_id}


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    q = _runs.get(run_id)
    if not q:
        raise HTTPException(status_code=404, detail="Unknown run_id")

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
            _runs.pop(run_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AGENT_PORT", "8099"))
    host = os.environ.get("AGENT_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, log_level="info")
