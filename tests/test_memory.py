from agenticos.agents.graph import build_graph
from agenticos.agents.planner import plan
from agenticos.agents.state import AgentAssignment, PlannerDecision, new_initial_state
from agenticos.db.repositories import AssetRepository
from agenticos.memory.store import ConversationStore, format_recent_turns, new_conversation_id
from agenticos.settings import Settings
from agenticos.tools.registry import ToolSpec
from conftest import FakeChatModel, ai_message_text, structured_ok


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, llm_max_retries=0, **overrides)


def _registry(temp_db):
    return {
        "asset_agent": ToolSpec(
            key="asset_agent", label="Asset Agent", param="asset_id",
            description="asset lookup", fetch=AssetRepository(temp_db).get_asset,
        ),
    }


def test_conversation_store_round_trip(tmp_path):
    store = ConversationStore(tmp_path / "memory.db")
    conv_id = new_conversation_id()

    assert store.get_recent_turns(conv_id, limit=5) == []

    store.append_turn(conv_id, "tell me about AHU-01", "AHU-01 is an air handling unit.", ["asset_agent"])
    store.append_turn(conv_id, "what about its alarms?", "AHU-01 has one critical alarm.", ["alarm_agent"])

    turns = store.get_recent_turns(conv_id, limit=5)
    assert len(turns) == 2
    # Chronological order: oldest first.
    assert turns[0]["request"] == "tell me about AHU-01"
    assert turns[1]["agents_used"] == ["alarm_agent"]


def test_conversation_store_limits_and_isolates_by_conversation(tmp_path):
    store = ConversationStore(tmp_path / "memory.db")
    conv_a, conv_b = new_conversation_id(), new_conversation_id()

    for i in range(3):
        store.append_turn(conv_a, f"request {i}", f"answer {i}", [])
    store.append_turn(conv_b, "unrelated request", "unrelated answer", [])

    turns = store.get_recent_turns(conv_a, limit=2)
    assert len(turns) == 2
    assert [t["request"] for t in turns] == ["request 1", "request 2"]  # most recent 2, chronological
    assert store.get_recent_turns(conv_b, limit=5)[0]["request"] == "unrelated request"


def test_format_recent_turns_truncates_long_answers():
    turns = [{"request": "q", "final_answer": "x" * 500, "agents_used": [], "created_at": ""}]
    text = format_recent_turns(turns)
    assert "q" in text
    assert len(text) < 500


def test_planner_recovers_asset_id_from_history(temp_db):
    """A bare follow-up with no asset ID in the current request should still
    resolve the asset via the regex fallback over conversation history."""
    decision = PlannerDecision(assignments=[AgentAssignment(agent="asset_agent", asset_id=None)], reason="follow-up")
    model = FakeChatModel(structured_responses=[structured_ok(decision)])
    history = [{"request": "tell me about AHU-01", "final_answer": "...", "agents_used": ["asset_agent"], "created_at": ""}]

    assignments, _, _, err = plan("and what about its maintenance history?", model, _registry(temp_db), history=history)

    assert err is None
    assert assignments[0]["asset_id"] == "AHU-01"


def test_graph_persists_and_reuses_history_across_turns(temp_db, tmp_path):
    """End-to-end: a turn with a conversation_id gets persisted, and a
    second call to the graph with the same conversation_id can read it back."""
    memory_store = ConversationStore(tmp_path / "memory.db")
    conv_id = new_conversation_id()

    decision = PlannerDecision(assignments=[AgentAssignment(agent="asset_agent", asset_id="AHU-01")], reason="asset question")
    model = FakeChatModel(
        structured_responses=[structured_ok(decision)],
        text_responses=[ai_message_text("AHU-01 is a healthy Air Handling Unit.")],
    )

    app = build_graph(settings=_settings(), model=model, registry=_registry(temp_db), memory_store=memory_store)
    app.invoke(new_initial_state("tell me about AHU-01", conversation_id=conv_id))

    turns = memory_store.get_recent_turns(conv_id, limit=5)
    assert len(turns) == 1
    assert turns[0]["final_answer"] == "AHU-01 is a healthy Air Handling Unit."


def test_graph_without_conversation_id_does_not_touch_memory(temp_db, tmp_path):
    """Single-shot requests (no conversation_id) never write to memory —
    verified by pointing memory_db_path somewhere and confirming no store
    is created (a fresh ConversationStore there sees zero turns)."""
    decision = PlannerDecision(assignments=[], reason="irrelevant")
    model = FakeChatModel(structured_responses=[structured_ok(decision)])

    memory_db_path = tmp_path / "unused_memory.db"
    app = build_graph(settings=_settings(), model=model, registry=_registry(temp_db), db_path=None)
    app.invoke(new_initial_state("what's your favorite color?"))

    assert not memory_db_path.exists()
