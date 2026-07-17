"""
Repositories over the SQLite-backed operational data (assets, alarms,
energy readings).

Each repository owns exactly one concern, takes an injectable `db_path` (so
tests can point at a temp file), and raises `ToolExecutionError` rather than
leaking raw `sqlite3.Error` up into the graph — the graph only needs to know
"this data source failed", not which driver it uses.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from agenticos.db.connection import session
from agenticos.exceptions import ToolExecutionError
from agenticos.logging_config import get_logger

logger = get_logger("db.repositories")


class AssetRepository:
    """Reads asset metadata from the `assets` table."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = db_path

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        try:
            with session(self._db_path) as conn:
                row = conn.execute(
                    "SELECT asset_id, type, install_date, last_maintenance, location "
                    "FROM assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("asset lookup failed", extra={"extra_fields": {"asset_id": asset_id, "error": str(exc)}})
            raise ToolExecutionError(f"Database error while looking up asset '{asset_id}': {exc}") from exc

        if row is None:
            return {"asset_id": asset_id, "found": False, "error": f"No asset found with ID '{asset_id}'."}
        return {"found": True, **dict(row)}


class AlarmRepository:
    """Reads recent alarm history from the `alarms` table."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = db_path

    def get_alarm_history(self, asset_id: str, limit: int = 5) -> dict[str, Any]:
        try:
            with session(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT alarm_type AS type, severity, triggered_at "
                    "FROM alarms WHERE asset_id = ? "
                    "ORDER BY triggered_at DESC LIMIT ?",
                    (asset_id, limit),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("alarm lookup failed", extra={"extra_fields": {"asset_id": asset_id, "error": str(exc)}})
            raise ToolExecutionError(f"Database error while looking up alarms for '{asset_id}': {exc}") from exc

        return {"asset_id": asset_id, "alarms": [dict(row) for row in rows]}


class EnergyRepository:
    """Reads energy telemetry from the `energy` table."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = db_path

    def get_energy(self, asset_id: str) -> dict[str, Any]:
        try:
            with session(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT voltage, current, power, consumption FROM energy WHERE asset_id = ?",
                    (asset_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("energy lookup failed", extra={"extra_fields": {"asset_id": asset_id, "error": str(exc)}})
            raise ToolExecutionError(f"Database error while looking up energy data for '{asset_id}': {exc}") from exc

        if not rows:
            return {"asset_id": asset_id, "readings": [], "error": f"No energy data found for '{asset_id}'."}
        return {"asset_id": asset_id, "readings": [dict(row) for row in rows]}
