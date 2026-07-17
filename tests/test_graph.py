from agenticos.agents.graph import build_graph
from agenticos.agents.state import AgentAssignment, PlannerDecision, new_initial_state
from agenticos.db.repositories import AssetRepository
from agenticos.settings import Settings
from agenticos.tools.registry import ToolSpec
from conftest import FakeChatModel, ai_message_text, structured_ok


def _settings() -> Settings:
    return Settings(_env_file=None, llm_max_retries=0)


def _registry(temp_db):
    return {
        "asset_agent": ToolSpec(
            key="asset_agent", label="Asset Agent", param="asset_id",
            description="asset lookup", fetch=AssetRepository(temp_db).get_asset,
        ),
    }


def test_full_graph_happy_path_makes_exactly_two_llm_calls(temp_db):
    """End-to-end: planner picks asset_agent, the specialist fetches data
    with zero LLM calls, supervisor synthesizes. Exactly 2 scripted LLM
    responses are consumed total — proof the graph never exceeds 2 calls
    regardless of how many specialists run."""
    planner_decision = PlannerDecision(
        assignments=[AgentAssignment(agent="asset_agent", asset_id="AHU-01")],
        reason="asset question",
    )
    model = FakeChatModel(
        structured_responses=[structured_ok(planner_decision)],
        text_responses=[ai_message_text("Summary: AHU-01 is a healthy Air Handling Unit.")],
    )

    app = build_graph(settings=_settings(), model=model, registry=_registry(temp_db))
    result = app.invoke(new_initial_state("tell me about AHU-01"))

    assert result["final_answer"] == "Summary: AHU-01 is a healthy Air Handling Unit."
    assert result["assignments"][0]["agent"] == "asset_agent"
    assert "asset_agent" in result["agent_results"]
    assert not model._structured_responses  # exactly one structured (planner) call consumed
    assert not model._text_responses  # exactly one text (supervisor) call consumed


def test_full_graph_no_agents_path(temp_db):
    """Planner selects nothing -> graph reaches END via the no_data fallback
    instead of dead-ending with an empty final_answer, and the supervisor
    (and its LLM call) never runs."""
    planner_decision = PlannerDecision(assignments=[], reason="request is unrelated to any agent")
    model = FakeChatModel(structured_responses=[structured_ok(planner_decision)])

    app = build_graph(settings=_settings(), model=model, registry=_registry(temp_db))
    result = app.invoke(new_initial_state("what's your favorite color?"))

    assert result["final_answer"] != ""
    assert "no investigation was run" in result["final_answer"]


def test_full_graph_planner_error_falls_back_gracefully(temp_db):
    model = FakeChatModel(structured_responses=[RuntimeError("Ollama not reachable")])

    app = build_graph(settings=_settings(), model=model, registry=_registry(temp_db))
    result = app.invoke(new_initial_state("tell me about AHU-01"))

    assert "no investigation was run" in result["final_answer"]
    assert any("Error calling model" in log for log in result["logs"])
