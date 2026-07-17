"""
Specialist "agents" — deterministic data-fetch functions, not LLM calls.

The original prototype had each specialist run its own tool-calling loop:
ask the LLM to decide to call its one bound tool, execute it, ask the LLM
again to summarize the raw result into prose. Both LLM calls were pure
overhead: the agent is *already* bound to exactly one tool by the time it
runs (the planner picked it), so "deciding" to call it is a formality, and
summarizing a small JSON blob into a paragraph the Supervisor is about to
read and re-synthesize anyway just adds latency without adding information.

Here, each assignment from the Planner is executed as a direct function
call against a repository (SQLite or the vector store), and the raw
structured result is handed straight to the Supervisor, which is the one
place actual synthesis needs to happen. Independent fetches run
concurrently (via a thread pool, since the underlying calls are blocking
I/O) when `settings.specialist_concurrency` is enabled.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agenticos.agents.state import AgentState
from agenticos.exceptions import ToolExecutionError
from agenticos.logging_config import get_logger
from agenticos.settings import Settings, get_settings
from agenticos.tools.registry import AgentKey, ToolSpec

logger = get_logger("agents.specialists")


def _run_one(spec: ToolSpec, assignment: dict[str, Any]) -> tuple[AgentKey, dict[str, Any] | None, str | None, float]:
    start = time.time()
    param_value = assignment.get(spec.param)
    try:
        result = spec.fetch(param_value)
        elapsed = round(time.time() - start, 3)
        return spec.key, result, None, elapsed
    except ToolExecutionError as exc:
        elapsed = round(time.time() - start, 3)
        logger.warning(
            "specialist fetch failed",
            extra={"extra_fields": {"agent": spec.key, "param": param_value, "error": str(exc)}},
        )
        return spec.key, None, str(exc), elapsed
    except Exception as exc:  # unexpected errors still shouldn't crash the whole graph
        elapsed = round(time.time() - start, 3)
        logger.exception("unexpected error in specialist fetch", extra={"extra_fields": {"agent": spec.key}})
        return spec.key, None, f"Unexpected error: {exc}", elapsed


def run_specialists_node(
    state: AgentState,
    registry: dict[AgentKey, ToolSpec],
    settings: Settings | None = None,
) -> dict:
    """Executes every planner-selected assignment. No LLM calls happen here."""
    settings = settings or get_settings()
    assignments = state.get("assignments", [])

    agent_results: dict[str, Any] = {}
    agent_errors: dict[str, str] = {}
    timings: dict[str, float] = {}
    logs = list(state.get("logs", []))

    jobs = [(registry[a["agent"]], a) for a in assignments if a["agent"] in registry]

    if settings.specialist_concurrency and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            results = list(pool.map(lambda job: _run_one(*job), jobs))
    else:
        results = [_run_one(spec, assignment) for spec, assignment in jobs]

    for agent_key, result, err, elapsed in results:
        timings[agent_key] = elapsed
        if err:
            agent_errors[agent_key] = err
            logs.append(f"[{agent_key}] Failed: {err}")
        else:
            agent_results[agent_key] = result
            logs.append(f"[{agent_key}] Retrieved data in {elapsed}s.")

    return {
        "agent_results": {**state.get("agent_results", {}), **agent_results},
        "agent_errors": {**state.get("agent_errors", {}), **agent_errors},
        "logs": logs,
        "timings": {**state.get("timings", {}), **timings},
    }
