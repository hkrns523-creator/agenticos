"""SQLite-backed conversation history.

Stores one row per completed turn (request + final answer + which agents
ran) keyed by `conversation_id`. This is deliberately simple — no summarization,
no embeddings — because the consumer is the Planner/Supervisor prompts,
which only need the last handful of turns verbatim to resolve pronouns and
follow-up references ("what about its alarms?"). If conversations grow long
enough that raw recent-turn context stops being enough, that's a summarization
layer to add on top of this store, not a reason to complicate it now.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from agenticos.db.connection import session
from agenticos.exceptions import ConversationMemoryError
from agenticos.logging_config import get_logger

logger = get_logger("memory.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    request TEXT NOT NULL,
    final_answer TEXT NOT NULL,
    agents_used TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation_id
    ON conversation_turns (conversation_id, id);
"""


class Turn(TypedDict):
    request: str
    final_answer: str
    agents_used: list[str]
    created_at: str


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def format_recent_turns(turns: list[Turn]) -> str:
    """Renders recent turns as compact prior-turn context for the
    Planner/Supervisor prompts. Kept short on purpose (request + a truncated
    answer) since this rides along on every call and isn't the thing being
    graded — it just needs to be enough for pronoun/entity resolution across
    turns, not a full transcript."""
    if not turns:
        return ""
    lines = []
    for turn in turns:
        answer = turn["final_answer"]
        if len(answer) > 300:
            answer = answer[:300] + "..."
        lines.append(f"User: {turn['request']}\nAssistant: {answer}")
    return "\n\n".join(lines)


class ConversationStore:
    """Thin repository over the `conversation_turns` table. Takes an
    injectable `db_path` (same pattern as the operational-data repositories)
    so tests can point it at a temp file."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            with session(self._db_path) as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("failed to initialize conversation memory schema", extra={"extra_fields": {"error": str(exc)}})
            raise ConversationMemoryError(f"Could not initialize conversation memory: {exc}") from exc

    def append_turn(self, conversation_id: str, request: str, final_answer: str, agents_used: list[str]) -> None:
        try:
            with session(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO conversation_turns (conversation_id, request, final_answer, agents_used, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, request, final_answer, json.dumps(agents_used), datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
        except sqlite3.Error as exc:
            # Memory is best-effort: a write failure shouldn't take down an
            # otherwise-successful investigation, so this is logged, not raised.
            logger.warning(
                "failed to persist conversation turn",
                extra={"extra_fields": {"conversation_id": conversation_id, "error": str(exc)}},
            )

    def get_recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        if limit <= 0:
            return []
        try:
            with session(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT request, final_answer, agents_used, created_at FROM conversation_turns "
                    "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                    (conversation_id, limit),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning(
                "failed to read conversation history",
                extra={"extra_fields": {"conversation_id": conversation_id, "error": str(exc)}},
            )
            return []

        turns: list[Turn] = [
            {
                "request": row["request"],
                "final_answer": row["final_answer"],
                "agents_used": json.loads(row["agents_used"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        turns.reverse()  # chronological order (oldest of the window first)
        return turns
