"""
Shared fixtures.

`FakeChatModel` mimics the subset of a LangChain chat model's interface the
app relies on (`invoke`, `with_structured_output(...).invoke`), returning
scripted responses instead of calling a real LLM — so the whole suite runs
offline, no Ollama server required.
"""
from __future__ import annotations

import sqlite3

import pytest
from langchain_core.messages import AIMessage


def ai_message_text(content: str, tokens=(10, 5, 15)) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": tokens[2]},
    )


class _FakeStructuredRunnable:
    def __init__(self, responses: list):
        self._responses = responses

    def invoke(self, messages):
        if not self._responses:
            raise AssertionError("FakeChatModel ran out of scripted structured responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    """`structured_responses` items are either an Exception, or a dict shaped
    like LangChain's `with_structured_output(..., include_raw=True)` output:
    `{"raw": AIMessage, "parsed": <schema instance>, "parsing_error": None}`.
    `text_responses` items are either an Exception or an AIMessage."""

    def __init__(self, structured_responses=None, text_responses=None):
        self._structured_responses = list(structured_responses or [])
        self._text_responses = list(text_responses or [])

    def with_structured_output(self, schema, include_raw=False):
        return _FakeStructuredRunnable(self._structured_responses)

    def invoke(self, messages):
        if not self._text_responses:
            raise AssertionError("FakeChatModel ran out of scripted text responses")
        response = self._text_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def structured_ok(parsed):
    return {"raw": ai_message_text("", tokens=(20, 10, 30)), "parsed": parsed, "parsing_error": None}


@pytest.fixture
def temp_db(tmp_path):
    """A throwaway SQLite DB with the same schema as production, seeded
    with one asset, two alarms, and one energy reading."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE assets (asset_id TEXT PRIMARY KEY, type TEXT, "
        "install_date TEXT, last_maintenance TEXT, location TEXT)"
    )
    conn.execute("CREATE TABLE alarms (asset_id TEXT, alarm_type TEXT, severity TEXT, triggered_at TEXT)")
    conn.execute("CREATE TABLE energy (asset_id TEXT, voltage REAL, current REAL, power REAL, consumption REAL)")
    conn.execute(
        "INSERT INTO assets VALUES ('AHU-01', 'Air Handling Unit', '2019-03-15', '2026-04-10', 'Building A')"
    )
    conn.executemany(
        "INSERT INTO alarms VALUES (?, ?, ?, ?)",
        [
            ("AHU-01", "High Temp", "Critical", "2026-07-09 12:00"),
            ("AHU-01", "Filter Pressure High", "Warning", "2026-07-07 12:00"),
        ],
    )
    conn.execute("INSERT INTO energy VALUES ('AHU-01', 440.0, 10.2, 4.5, 250.0)")
    conn.commit()
    conn.close()
    return str(db_path)
