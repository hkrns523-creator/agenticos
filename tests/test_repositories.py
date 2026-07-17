import pytest

from agenticos.db.repositories import AlarmRepository, AssetRepository, EnergyRepository
from agenticos.exceptions import ToolExecutionError


def test_asset_found(temp_db):
    repo = AssetRepository(temp_db)
    result = repo.get_asset("AHU-01")
    assert result["found"] is True
    assert result["type"] == "Air Handling Unit"


def test_asset_not_found(temp_db):
    repo = AssetRepository(temp_db)
    result = repo.get_asset("DOES-NOT-EXIST")
    assert result["found"] is False
    assert "DOES-NOT-EXIST" in result["error"]


def test_asset_db_error():
    repo = AssetRepository("/nonexistent-dir/nope.db")
    with pytest.raises(ToolExecutionError):
        repo.get_asset("AHU-01")


def test_alarm_history_ordered_and_limited(temp_db):
    repo = AlarmRepository(temp_db)
    result = repo.get_alarm_history("AHU-01", limit=1)
    assert result["asset_id"] == "AHU-01"
    assert len(result["alarms"]) == 1
    assert result["alarms"][0]["type"] == "High Temp"  # most recent first


def test_alarm_history_db_error():
    repo = AlarmRepository("/nonexistent-dir/nope.db")
    with pytest.raises(ToolExecutionError):
        repo.get_alarm_history("AHU-01")


def test_energy_found(temp_db):
    repo = EnergyRepository(temp_db)
    result = repo.get_energy("AHU-01")
    assert result["readings"][0]["power"] == 4.5


def test_energy_not_found(temp_db):
    repo = EnergyRepository(temp_db)
    result = repo.get_energy("NOPE-99")
    assert result["readings"] == []
    assert "error" in result
