"""Shared state and structured-output schemas for the graph."""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from agenticos.tools.registry import AGENT_KEYS, AgentKey


class AgentAssignment(BaseModel):
    """One specialist agent the planner has decided to run, plus whatever
    parameter it needs extracted from the user's request up front — this is
    what lets specialists skip their own "decide what to call" LLM call."""

    agent: Literal[AGENT_KEYS] = Field(description="Which specialist agent to run.")  # type: ignore[valid-type]
    asset_id: Optional[str] = Field(
        default=None,
        description="The asset ID mentioned in the request (e.g. 'AHU-01', 'CH-01'), "
        "required for asset_agent, alarm_agent, and energy_agent.",
    )
    topic: Optional[str] = Field(
        default=None,
        description="The topic or question to look up documentation for, "
        "required for documentation_agent.",
    )


class PlannerDecision(BaseModel):
    """Structured output the planner LLM call must produce. Doing agent
    selection and parameter extraction in one shot (instead of one call to
    pick agents, then a second call per agent to decide how to call its
    tool) is what collapses the graph from ~10 possible LLM calls to 2."""

    reason: str = Field(
        description="Step-by-step: which agents are relevant to the request and why, decided before the final assignment list."
    )
    assignments: list[AgentAssignment] = Field(
        default_factory=list,
        description="The specialist agents to run for this request, each with the parameters it needs.",
    )


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AgentState(TypedDict, total=False):
    request: str
    conversation_id: str
    history: list[dict[str, Any]]  # recent prior turns: {request, final_answer, agents_used, created_at}
    assignments: list[dict[str, Any]]  # serialized AgentAssignment objects
    agent_results: dict[AgentKey, dict[str, Any]]
    agent_errors: dict[AgentKey, str]
    final_answer: str
    logs: list[str]
    timings: dict[str, float]
    token_usage: TokenUsage


def empty_token_usage() -> TokenUsage:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def add_token_usage(existing: TokenUsage, new: TokenUsage) -> TokenUsage:
    return {
        "input_tokens": existing.get("input_tokens", 0) + new.get("input_tokens", 0),
        "output_tokens": existing.get("output_tokens", 0) + new.get("output_tokens", 0),
        "total_tokens": existing.get("total_tokens", 0) + new.get("total_tokens", 0),
    }


def extract_token_usage(raw_message: Any) -> TokenUsage:
    """Safely reads token usage from a raw AIMessage, if the provider reports it."""
    usage = getattr(raw_message, "usage_metadata", None) if raw_message else None
    if usage:
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    return empty_token_usage()


def new_initial_state(request: str, conversation_id: str = "", history: list[dict[str, Any]] | None = None) -> AgentState:
    return {
        "request": request,
        "conversation_id": conversation_id,
        "history": history or [],
        "assignments": [],
        "agent_results": {},
        "agent_errors": {},
        "final_answer": "",
        "logs": [],
        "timings": {},
        "token_usage": empty_token_usage(),
    }
