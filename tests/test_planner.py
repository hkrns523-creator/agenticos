from agenticos.agents.planner import plan, planner_node
from agenticos.agents.state import AgentAssignment, PlannerDecision
from agenticos.db.repositories import AssetRepository
from agenticos.tools.registry import ToolSpec
from conftest import FakeChatModel, ai_message_text, structured_ok


def _registry(temp_db):
    return {
        "asset_agent": ToolSpec(
            key="asset_agent", label="Asset Agent", param="asset_id",
            description="asset lookup", fetch=AssetRepository(temp_db).get_asset,
        ),
        "documentation_agent": ToolSpec(
            key="documentation_agent", label="Documentation Agent", param="topic",
            description="doc lookup", fetch=lambda topic: {"topic": topic, "documents": []},
        ),
    }


def test_plan_happy_path(temp_db):
    decision = PlannerDecision(
        assignments=[AgentAssignment(agent="asset_agent", asset_id="AHU-01")],
        reason="asks about an asset",
    )
    model = FakeChatModel(structured_responses=[structured_ok(decision)])

    assignments, log_msg, tokens, err = plan("tell me about AHU-01", model, _registry(temp_db))

    assert err is None
    assert assignments == [{"agent": "asset_agent", "asset_id": "AHU-01", "topic": None}]
    assert "asset_agent" in log_msg
    assert tokens["total_tokens"] == 30


def test_plan_fills_missing_asset_id_via_regex(temp_db):
    """Model picked the right agent but forgot to extract the asset_id —
    the regex safety net should recover it from the request text."""
    decision = PlannerDecision(
        assignments=[AgentAssignment(agent="asset_agent", asset_id=None)],
        reason="asset question",
    )
    model = FakeChatModel(structured_responses=[structured_ok(decision)])

    assignments, _, _, err = plan("What's up with AHU-01?", model, _registry(temp_db))

    assert err is None
    assert assignments[0]["asset_id"] == "AHU-01"


def test_plan_drops_assignment_with_unrecoverable_asset_id(temp_db):
    decision = PlannerDecision(
        assignments=[AgentAssignment(agent="asset_agent", asset_id=None)],
        reason="vague",
    )
    model = FakeChatModel(structured_responses=[structured_ok(decision)])

    assignments, _, _, err = plan("something is wrong somewhere", model, _registry(temp_db))

    assert assignments == []


def test_plan_documentation_falls_back_to_full_request(temp_db):
    decision = PlannerDecision(
        assignments=[AgentAssignment(agent="documentation_agent", topic=None)],
        reason="procedural question",
    )
    model = FakeChatModel(structured_responses=[structured_ok(decision)])

    assignments, _, _, err = plan("what's the SOP for filter replacement?", model, _registry(temp_db))

    assert assignments[0]["topic"] == "what's the SOP for filter replacement?"


def test_plan_model_error(temp_db):
    model = FakeChatModel(structured_responses=[RuntimeError("Ollama not reachable")])

    assignments, log_msg, tokens, err = plan("anything", model, _registry(temp_db))

    assert assignments == []
    assert err is not None
    assert "Error calling model" in log_msg


def test_planner_node_updates_state(temp_db):
    decision = PlannerDecision(
        assignments=[AgentAssignment(agent="asset_agent", asset_id="AHU-01")],
        reason="asset question",
    )
    model = FakeChatModel(structured_responses=[structured_ok(decision)])

    result = planner_node({"request": "tell me about AHU-01", "logs": []}, model, _registry(temp_db))

    assert result["assignments"][0]["agent"] == "asset_agent"
    assert "planner" in result["timings"]
    assert result["token_usage"]["total_tokens"] == 30
