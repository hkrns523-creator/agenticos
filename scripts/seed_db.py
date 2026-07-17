"""
Seeds a local SQLite database with realistic building-operations data:
assets, alarm history, and energy telemetry.

Run this once to create/refresh data/agenticos.db:

    python scripts/seed_db.py

Note: the original seed script only created `assets` and `alarms` — the
`energy` table that `EnergyRepository` queries was missing here even though
it existed in the shipped database file. That's fixed below so a fresh
clone can reproduce the full dataset from scratch.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from agenticos.settings import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    install_date TEXT NOT NULL,
    last_maintenance TEXT NOT NULL,
    location TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    alarm_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    triggered_at TEXT NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS energy (
    asset_id TEXT NOT NULL,
    voltage REAL NOT NULL,
    current REAL NOT NULL,
    power REAL NOT NULL,
    consumption REAL NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);
"""

ASSETS = [
    ("AHU-01", "Air Handling Unit", "2019-03-15", "2026-04-10", "Building A, Rooftop"),
    ("AHU-02", "Air Handling Unit", "2020-07-22", "2026-05-02", "Building A, Rooftop"),
    ("CH-01", "Centrifugal Chiller", "2017-11-01", "2026-02-18", "Building A, Basement Plant Room"),
    ("PMP-04", "Circulation Pump", "2021-01-10", "2026-06-01", "Building B, Mechanical Room"),
]


def _days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d %H:%M")


ALARMS = [
    ("AHU-01", "High Supply Air Temp", "Critical", _days_ago(0)),
    ("AHU-01", "Filter Differential Pressure High", "Warning", _days_ago(2)),
    ("AHU-01", "Fan Belt Slip Detected", "Warning", _days_ago(14)),
    ("AHU-02", "Low Supply Air Temp", "Warning", _days_ago(5)),
    ("CH-01", "Condenser Water Flow Low", "Critical", _days_ago(1)),
    ("CH-01", "Refrigerant Pressure High", "Warning", _days_ago(9)),
    ("PMP-04", "Vibration Threshold Exceeded", "Warning", _days_ago(3)),
]

ENERGY = [
    ("AHU-01", 440.0, 10.2, 4.5, 250.0),
    ("BOILER-01", 415.0, 18.5, 7.8, 620.0),
]


def seed(db_path: str | None = None, if_missing: bool = False) -> None:
    path = db_path or str(get_settings().db_path)

    if if_missing and Path(path).exists():
        conn = sqlite3.connect(path)
        try:
            has_data = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0] > 0
        except sqlite3.OperationalError:
            has_data = False  # table doesn't exist yet
        finally:
            conn.close()
        if has_data:
            print(f"{path} already has data; skipping seed (--if-missing).")
            return

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    cur.execute("DELETE FROM assets")
    cur.execute("DELETE FROM alarms")
    cur.execute("DELETE FROM energy")

    cur.executemany(
        "INSERT INTO assets (asset_id, type, install_date, last_maintenance, location) VALUES (?, ?, ?, ?, ?)",
        ASSETS,
    )
    cur.executemany(
        "INSERT INTO alarms (asset_id, alarm_type, severity, triggered_at) VALUES (?, ?, ?, ?)",
        ALARMS,
    )
    cur.executemany(
        "INSERT INTO energy (asset_id, voltage, current, power, consumption) VALUES (?, ?, ?, ?, ?)",
        ENERGY,
    )

    conn.commit()
    conn.close()
    print(f"Seeded {len(ASSETS)} assets, {len(ALARMS)} alarms, {len(ENERGY)} energy readings into {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the AgenticOS SQLite database.")
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Skip seeding if the database already exists and has data (safe for repeated container starts).",
    )
    args = parser.parse_args()
    seed(if_missing=args.if_missing)
