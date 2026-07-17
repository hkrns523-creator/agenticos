"""
Single source of truth for "what specialist data sources exist".

The original prototype had four near-identical node functions
(`alarm_node`, `asset_node`, `documentation_node`, `energy_node`), each
wrapping an LLM tool-calling loop around one function. Now that specialist
execution is deterministic (see agents/specialists.py), there's no need for
per-agent node functions at all: this registry maps an agent key to (a) the
parameter it needs and (b) the plain callable that fetches its data. Both
the planner's structured-output schema and the specialist executor read
from this same registry, so adding a fifth data source means adding one
entry here — not touching the planner prompt, a new node function, and the
graph wiring separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from agenticos.db.repositories import AlarmRepository, AssetRepository, EnergyRepository
from agenticos.rag.documentation_repository import DocumentationRepository
from agenticos.settings import Settings

AgentKey = Literal["asset_agent", "alarm_agent", "documentation_agent", "energy_agent"]
ParamName = Literal["asset_id", "topic"]

AGENT_KEYS: tuple[AgentKey, ...] = ("asset_agent", "alarm_agent", "documentation_agent", "energy_agent")


@dataclass(frozen=True)
class ToolSpec:
    key: AgentKey
    label: str
    param: ParamName
    description: str  # shown to the planner LLM so it knows when to pick this agent
    fetch: Callable[[str], dict[str, Any]]


def build_registry(settings: Settings, db_path: str | Path | None = None) -> dict[AgentKey, ToolSpec]:
    """Builds the live registry, wiring each spec's `fetch` to a repository
    instance. `db_path` is exposed separately so tests can point the SQLite
    repositories at a temp DB without touching the vector store paths.

    When `settings.rag_enabled` is False, `documentation_agent` is omitted
    entirely rather than constructed-then-ignored: `DocumentationRepository`
    loads the embedding model (sentence-transformers -> torch) eagerly in
    its constructor, which is a few hundred MB of RAM that a
    memory-constrained deployment (e.g. a free-tier host) may not have.
    """
    asset_repo = AssetRepository(db_path or settings.db_path)
    alarm_repo = AlarmRepository(db_path or settings.db_path)
    energy_repo = EnergyRepository(db_path or settings.db_path)

    registry: dict[AgentKey, ToolSpec] = {
        "asset_agent": ToolSpec(
            key="asset_agent",
            label="Asset Agent",
            param="asset_id",
            description="Looks up asset metadata: type, install date, last maintenance, location.",
            fetch=asset_repo.get_asset,
        ),
        "alarm_agent": ToolSpec(
            key="alarm_agent",
            label="Alarm Agent",
            param="asset_id",
            description="Looks up recent alarm/fault history for an asset.",
            fetch=alarm_repo.get_alarm_history,
        ),
        "energy_agent": ToolSpec(
            key="energy_agent",
            label="Energy Agent",
            param="asset_id",
            description="Looks up recent energy telemetry (voltage, current, power, consumption) for an asset.",
            fetch=energy_repo.get_energy,
        ),
    }

    if settings.rag_enabled:
        doc_repo = DocumentationRepository(
            docs_dir=settings.docs_dir,
            vector_db_dir=settings.vector_db_dir,
            embedding_model=settings.embedding_model,
            top_k=settings.rag_top_k,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            search_type=settings.rag_search_type,
            fetch_k=settings.rag_fetch_k,
            score_threshold=settings.rag_score_threshold,
        )
        registry["documentation_agent"] = ToolSpec(
            key="documentation_agent",
            label="Documentation Agent",
            param="topic",
            description="Looks up relevant standard operating procedures (SOPs) and manual excerpts for a topic.",
            fetch=doc_repo.get_sop,
        )

    return registry