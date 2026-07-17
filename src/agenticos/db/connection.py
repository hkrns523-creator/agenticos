"""SQLite connection management.

A single `session()` context manager, parameterized by path so tests can
point it at a throwaway file without monkeypatching module-level globals
scattered across the codebase.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agenticos.settings import get_settings


@contextmanager
def session(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with row access by column name, closed
    automatically (even on error) when the block exits."""
    path = db_path or get_settings().db_path
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
