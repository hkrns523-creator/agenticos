from agenticos.agents.supervisor import no_data_node, supervisor_node
from agenticos.settings import Settings
from conftest import FakeChatModel, ai_message_text


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_supervisor_happy_path():
    model = FakeChatModel(text_responses=[ai_message_text("Final synthesized answer.")])
    state = {
        "request": "what's going on with AHU-01?",
        "agent_results": {"asset_agent": {"type": "Air Handling Unit"}},
        "agent_errors": {},
        "logs": [],
        "timings": {},
        "token_usage": {},
    }

    result = supervisor_node(state, model, _settings())

    assert result["final_answer"] == "Final synthesized answer."
    assert "supervisor" in result["timings"]
    assert result["token_usage"]["total_tokens"] == 15


def test_supervisor_model_error_falls_back_to_raw_findings():
    model = FakeChatModel(text_responses=[RuntimeError("offline")])
    state = {
        "request": "what's going on?",
        "agent_results": {"asset_agent": {"type": "Air Handling Unit"}},
        "agent_errors": {},
        "logs": [],
        "timings": {},
        "token_usage": {},
    }

    result = supervisor_node(state, model, _settings(llm_max_retries=0))

    assert "couldn't synthesize" in result["final_answer"]
    assert "Air Handling Unit" in result["final_answer"]


def test_supervisor_surfaces_specialist_errors():
    model = FakeChatModel(text_responses=[ai_message_text("Synthesized despite one failure.")])
    state = {
        "request": "check AHU-01",
        "agent_results": {},
        "agent_errors": {"asset_agent": "Database error"},
        "logs": [],
        "timings": {},
        "token_usage": {},
    }

    result = supervisor_node(state, model, _settings())

    assert result["final_answer"] == "Synthesized despite one failure."


def test_no_data_node():
    result = no_data_node({"logs": []})
    assert result["final_answer"]
    assert "no investigation was run" in result["final_answer"]
    assert any("No agents were selected" in log for log in result["logs"])
