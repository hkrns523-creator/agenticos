"""
Builds the LangGraph app.

Shape: planner -> (run_specialists | no_data) -> supervisor -> END

Only `planner` and `supervisor` make LLM calls — exactly 2 per request,
regardless of how many specialist data sources get consulted. Compare to
the original prototype's up to 10 (1 planner + 2 per specialist x 4 + 1
supervisor).
"""
from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from agenticos.agents.planner import planner_node
from agenticos.agents.specialists import run_specialists_node
from agenticos.agents.state import AgentState
from agenticos.agents.supervisor import no_data_node, supervisor_node
from agenticos.llm.client import build_chat_model
from agenticos.logging_config import get_logger
from agenticos.memory.store import ConversationStore
from agenticos.settings import Settings, get_settings
from agenticos.tools.registry import build_registry

logger = get_logger("agents.graph")


def _route_after_planner(state: AgentState) -> str:
    return "run_specialists" if state.get("assignments") else "no_data"


def _persist_memory_node(
    state: AgentState,
    memory_store: ConversationStore | None,
    memory_db_path,
    enabled: bool,
) -> dict:
    """Saves the completed turn (request + final answer + which agents ran)
    so a later request in the same conversation can be planned/synthesized
    with that context. No-op if memory is disabled or no conversation_id was
    provided (e.g. one-off CLI invocations) — this keeps single-shot requests
    (and tests, which never set conversation_id) from touching the memory
    database at all."""
    conversation_id = state.get("conversation_id")
    if not enabled or not conversation_id:
        return {}

    store = memory_store
    if store is None:
        try:
            store = ConversationStore(memory_db_path)
        except Exception:
            logger.warning("conversation memory unavailable; skipping persistence for this turn")
            return {}

    agents_used = [a["agent"] for a in state.get("assignments", [])]
    store.append_turn(conversation_id, state["request"], state.get("final_answer", ""), agents_used)
    return {}


def build_graph(
    settings: Settings | None = None,
    db_path: str | None = None,
    model=None,
    registry=None,
    memory_store: ConversationStore | None = None,
):
    """Builds and compiles the graph.

    `db_path` lets callers (tests, CLI flags) point the SQLite-backed
    specialists at a specific database file without mutating global
    settings. `model` lets tests inject a fake chat model instead of
    requiring a live Ollama server. `registry` lets tests inject fake
    ToolSpecs instead of building the (heavier) real RAG/DB-backed ones.
    `memory_store` lets tests/callers inject a ConversationStore pointed at
    a temp DB; when omitted and memory is enabled, one is built from settings.
    """
    settings = settings or get_settings()
    model = model or build_chat_model(settings)
    registry = registry if registry is not None else build_registry(settings, db_path=db_path)
    memory_db_path = db_path or settings.resolved_memory_db_path

    graph = StateGraph(AgentState)
    graph.add_node("planner", partial(planner_node, model=model, registry=registry))
    graph.add_node("run_specialists", partial(run_specialists_node, registry=registry, settings=settings))
    graph.add_node("no_data", no_data_node)
    graph.add_node("supervisor", partial(supervisor_node, model=model, settings=settings))
    graph.add_node(
        "persist_memory",
        partial(
            _persist_memory_node,
            memory_store=memory_store,
            memory_db_path=memory_db_path,
            enabled=settings.memory_enabled,
        ),
    )

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", _route_after_planner, ["run_specialists", "no_data"])
    graph.add_edge("run_specialists", "supervisor")
    graph.add_edge("supervisor", "persist_memory")
    graph.add_edge("no_data", "persist_memory")
    graph.add_edge("persist_memory", END)

    return graph.compile()
