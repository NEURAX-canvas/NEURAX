"""Client for `neurax-service`'s agent-memory endpoints (`agent_memory.rs`).

Three tiers, scoped by `project_id` alone — see `agent_memory.rs`'s own
module doc for why there is no user-level tier: `neurax-ui` has no real
Supabase auth integration to key one by, verified before any of this was
written rather than assumed.

- **Core** — a short list of stated preferences, injected into every
  `plan_step` call's prompt automatically (not a tool the model calls to
  *read*; `remember_preference` is the tool that *writes* one).
- **Archival** — past designs' rationale, searchable by `search_past_designs`
  (research/explanation modes), written automatically when a creation/
  research run finishes successfully.
- **Recall** — conversation continuity, read automatically at the start of
  a run with a `project_id` and appended to automatically at the end.

Every function here is best-effort: a project with no memory yet, a
`neurax-service` that isn't reachable, or a Supabase project that was never
provisioned with these tables (see `agent_memory.rs`'s doc comment for the
schema) all degrade to "no memory available" rather than failing the run
that asked for it. Memory is a real feature this loop now has; it is not
one any of it is allowed to depend on to function at all.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [NEURAX-MEMORY] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

NEURAX_SERVICE_URL = os.environ.get("NEURAX_SERVICE_URL", "http://127.0.0.1:9098")

_TIMEOUT_S = 10.0


async def get_core_preferences(project_id: str) -> list[str]:
    if not project_id:
        return []
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(f"{NEURAX_SERVICE_URL}/memory/core", params={"project_id": project_id})
            r.raise_for_status()
            return list(r.json().get("preferences") or [])
    except Exception as e:
        logger.warning(f"could not read core memory for project {project_id}: {e}")
        return []


async def add_core_preference(project_id: str, preference: str) -> bool:
    if not project_id or not preference:
        return False
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.post(
                f"{NEURAX_SERVICE_URL}/memory/core/preference",
                json={"project_id": project_id, "preference": preference},
            )
            r.raise_for_status()
            return True
    except Exception as e:
        logger.warning(f"could not write core memory for project {project_id}: {e}")
        return False


async def search_past_designs(project_id: str, query: str, limit: int = 5) -> list[str]:
    if not project_id:
        return []
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(
                f"{NEURAX_SERVICE_URL}/memory/archival",
                params={"project_id": project_id, "query": query, "limit": limit},
            )
            r.raise_for_status()
            return list(r.json().get("entries") or [])
    except Exception as e:
        logger.warning(f"could not search archival memory for project {project_id}: {e}")
        return []


async def add_archival_entry(project_id: str, content: str) -> bool:
    if not project_id or not content:
        return False
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.post(
                f"{NEURAX_SERVICE_URL}/memory/archival",
                json={"project_id": project_id, "content": content},
            )
            r.raise_for_status()
            return True
    except Exception as e:
        logger.warning(f"could not write archival memory for project {project_id}: {e}")
        return False


async def get_recent_conversation(project_id: str, limit: int = 8) -> list[dict[str, str]]:
    if not project_id:
        return []
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.get(
                f"{NEURAX_SERVICE_URL}/memory/conversation",
                params={"project_id": project_id, "limit": limit},
            )
            r.raise_for_status()
            return list(r.json().get("turns") or [])
    except Exception as e:
        logger.warning(f"could not read conversation memory for project {project_id}: {e}")
        return []


async def append_conversation_turns(project_id: str, turns: list[dict[str, str]]) -> bool:
    if not project_id or not turns:
        return False
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            r = await client.post(
                f"{NEURAX_SERVICE_URL}/memory/conversation",
                json={"project_id": project_id, "turns": turns},
            )
            r.raise_for_status()
            return True
    except Exception as e:
        logger.warning(f"could not write conversation memory for project {project_id}: {e}")
        return False
