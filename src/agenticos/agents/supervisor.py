"""
The Supervisor: the second and final LLM call in the graph.

Synthesizes the raw, structured results gathered by the specialists (now
plain dicts, not LLM-generated prose — see specialists.py) directly into
one final answer. Skipping the per-specialist summarization step means the
Supervisor is working from the same underlying data either way, just
without an extra model call and the risk of that call re-summarizing
information ever so slightly differently each time.
"""
from __future__ import annotations

import json
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agenticos.agents.state import AgentState, add_token_usage, extract_token_usage
from agenticos.llm.client import safe_invoke
from agenticos.memory.store import format_recent_turns
from agenticos.settings import Settings, get_settings

_SUPERVISOR_INSTRUCTIONS = (
    "Synthesize these findings into one clear, well-organized final answer for a "
    "building-operations engineer. Include: a brief summary, probable causes, and "
    "recommended next steps. If a specialist reported an error or found nothing, "
    "say so plainly instead of guessing. Do not restate the raw data verbatim — "
    "interpret it. If earlier conversation turns are provided, use them only to keep "
    "this answer consistent with what was already said, not as new findings."
)


def _format_context(agent_results: dict, agent_errors: dict) -> str:
    parts = []
    for agent, result in agent_results.items():
        parts.append(f"{agent} findings:\n{json.dumps(result, indent=2, default=str)}")
    for agent, error in agent_errors.items():
        parts.append(f"{agent} error: {error}")
    return "\n\n".join(parts) if parts else "No data was retrieved."


def supervisor_node(state: AgentState, model: BaseChatModel, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    start = time.time()

    context = _format_context(state.get("agent_results", {}), state.get("agent_errors", {}))
    history_text = format_recent_turns(state.get("history", []))
    history_block = f"\n\nEarlier turns in this conversation:\n\n{history_text}" if history_text else ""
    prompt = (
        f"A user asked: '{state['request']}'{history_block}\n\n"
        f"Specialist findings:\n\n{context}\n\n{_SUPERVISOR_INSTRUCTIONS}"
    )

    response, err = safe_invoke(model, [HumanMessage(prompt)], settings)
    logs = list(state.get("logs", []))

    if err:
        final_answer = (
            "I gathered data from the specialist agents but couldn't synthesize a final "
            f"answer because the model was unreachable ({err}). Raw findings:\n\n{context}"
        )
        tokens = extract_token_usage(None)
        logs.append(f"[Supervisor] Error calling model: {err}")
    else:
        final_answer = response.content
        tokens = extract_token_usage(response)
        logs.append("[Supervisor] Synthesized final answer from specialist findings.")

    elapsed = round(time.time() - start, 3)
    return {
        "final_answer": final_answer,
        "logs": logs,
        "timings": {**state.get("timings", {}), "supervisor": elapsed},
        "token_usage": add_token_usage(state.get("token_usage", {}), tokens),
    }


def no_data_node(state: AgentState) -> dict:
    """Fallback when the Planner selected no agents (irrelevant/ambiguous
    request, or a planner error) — keeps the graph from dead-ending with an
    empty final_answer."""
    logs = state.get("logs", []) + ["[Planner] No agents were selected; no investigation was run."]
    return {
        "final_answer": (
            "I wasn't able to determine which specialist agents should handle this "
            "request, so no investigation was run. Please try rephrasing your request "
            "with a specific asset ID (e.g. 'AHU-01') or topic."
        ),
        "logs": logs,
    }
