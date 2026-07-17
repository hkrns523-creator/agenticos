"""
Structured logging setup.

Production deployments want machine-parseable logs (JSON, one object per
line) that ship cleanly into log aggregators; local development wants
readable text. `configure_logging()` picks the formatter based on settings
and is idempotent (safe to call multiple times, e.g. once from the CLI
entrypoint and once from the Streamlit entrypoint).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from agenticos.settings import get_settings

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload)


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter() if settings.log_json else logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger("agenticos")
    root.setLevel(settings.log_level.upper())
    root.handlers = [handler]
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"agenticos.{name}")
