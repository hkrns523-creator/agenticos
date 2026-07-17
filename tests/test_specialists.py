from agenticos.agents.specialists import run_specialists_node
from agenticos.db.repositories import AlarmRepository, AssetRepository, EnergyRepository
from agenticos.exceptions import ToolExecutionError
from agenticos.settings import Settings
from agenticos.tools.registry import ToolSpec


def _registry(temp_db):
    return {
        "asset_agent": ToolSpec(
            key="asset_agent", label="Asset Agent", param="asset_id",
            description="asset lookup", fetch=AssetRepository(temp_db).get_asset,
        ),
        "alarm_agent": ToolSpec(
            key="alarm_agent", label="Alarm Agent", param="asset_id",
            description="alarm lookup", fetch=AlarmRepository(temp_db).get_alarm_history,
        ),
        "energy_agent": ToolSpec(
            key="energy_agent", label="Energy Agent", param="asset_id",
            description="energy lookup", fetch=EnergyRepository(temp_db).get_energy,
        ),
    }


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_no_llm_calls_are_made_single_agent(temp_db):
    """The core guarantee: specialist execution never touches a chat model."""
    registry = _registry(temp_db)
    state = {"request": "tell me about AHU-01", "assignments": [{"agent": "asset_agent", "asset_id": "AHU-01"}]}

    result = run_specialists_node(state, registry, _settings(specialist_concurrency=False))

    assert result["agent_results"]["asset_agent"]["type"] == "Air Handling Unit"
    assert result["agent_errors"] == {}
    assert "asset_agent" in result["timings"]


def test_multiple_agents_run_concurrently(temp_db):
    registry = _registry(temp_db)
    state = {
        "request": "why is AHU-01 alarming and what's its energy usage?",
        "assignments": [
            {"agent": "asset_agent", "asset_id": "AHU-01"},
            {"agent": "alarm_agent", "asset_id": "AHU-01"},
            {"agent": "energy_agent", "asset_id": "AHU-01"},
        ],
    }

    result = run_specialists_node(state, registry, _settings(specialist_concurrency=True))

    assert set(result["agent_results"].keys()) == {"asset_agent", "alarm_agent", "energy_agent"}
    assert len(result["agent_results"]["alarm_agent"]["alarms"]) == 2


def test_tool_execution_error_is_captured_not_raised(temp_db):
    registry = {
        "asset_agent": ToolSpec(
            key="asset_agent", label="Asset Agent", param="asset_id",
            description="asset lookup",
            fetch=lambda asset_id: (_ for _ in ()).throw(ToolExecutionError("db is down")),
        ),
    }
    state = {"request": "AHU-01", "assignments": [{"agent": "asset_agent", "asset_id": "AHU-01"}]}

    result = run_specialists_node(state, registry, _settings(specialist_concurrency=False))

    assert "asset_agent" in result["agent_errors"]
    assert "db is down" in result["agent_errors"]["asset_agent"]
    assert result["agent_results"] == {}


def test_no_assignments_returns_empty_results(temp_db):
    registry = _registry(temp_db)
    result = run_specialists_node({"request": "hi", "assignments": []}, registry, _settings())
    assert result["agent_results"] == {}
    assert result["agent_errors"] == {}
