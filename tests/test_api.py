from fastapi.testclient import TestClient

from agenticos.agents.graph import build_graph
from agenticos.agents.state import AgentAssignment, PlannerDecision
from agenticos.api import app as api_app
from agenticos.db.repositories import AssetRepository
from agenticos.memory.store import ConversationStore
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


def _client_with_fake_backend(temp_db, tmp_path, structured_responses, text_responses):
    """Builds a TestClient without running the real lifespan (which would try
    to reach Ollama and build the heavy RAG stack); instead injects a graph
    built from fakes directly into the module-level app state, the same
    pattern the other tests use via build_graph(model=..., registry=...)."""
    model = FakeChatModel(structured_responses=structured_responses, text_responses=text_responses)
    settings = _settings()
    memory_store = ConversationStore(tmp_path / "memory.db")
    graph = build_graph(settings=settings, model=model, registry=_registry(temp_db), memory_store=memory_store)

    api_app._state.clear()
    api_app._state.update({"app_graph": graph, "memory_store": memory_store, "settings": settings, "ready": True})
    return TestClient(api_app.app)


def test_health_always_ok():
    client = TestClient(api_app.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_is_503_before_startup():
    api_app._state.clear()
    api_app._state["ready"] = False
    client = TestClient(api_app.app)
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_investigate_happy_path(temp_db, tmp_path):
    decision = PlannerDecision(assignments=[AgentAssignment(agent="asset_agent", asset_id="AHU-01")], reason="asset question")
    client = _client_with_fake_backend(
        temp_db, tmp_path,
        structured_responses=[structured_ok(decision)],
        text_responses=[ai_message_text("AHU-01 is a healthy Air Handling Unit.")],
    )

    resp = client.post("/investigate", json={"request": "tell me about AHU-01", "conversation_id": "conv-1"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["final_answer"] == "AHU-01 is a healthy Air Handling Unit."
    assert body["assignments"][0]["agent"] == "asset_agent"
    assert body["conversation_id"] == "conv-1"


def test_investigate_persists_and_history_endpoint_returns_it(temp_db, tmp_path):
    decision = PlannerDecision(assignments=[AgentAssignment(agent="asset_agent", asset_id="AHU-01")], reason="asset question")
    client = _client_with_fake_backend(
        temp_db, tmp_path,
        structured_responses=[structured_ok(decision)],
        text_responses=[ai_message_text("AHU-01 is fine.")],
    )

    client.post("/investigate", json={"request": "tell me about AHU-01", "conversation_id": "conv-2"})
    resp = client.get("/conversations/conv-2/history")

    assert resp.status_code == 200
    turns = resp.json()["turns"]
    assert len(turns) == 1
    assert turns[0]["final_answer"] == "AHU-01 is fine."


def test_investigate_rejects_empty_request(temp_db, tmp_path):
    client = _client_with_fake_backend(temp_db, tmp_path, structured_responses=[], text_responses=[])
    resp = client.post("/investigate", json={"request": ""})
    assert resp.status_code == 422
