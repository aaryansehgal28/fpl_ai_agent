"""Logging helpers for consistent module-level loggers."""

from __future__ import annotations

import json
import logging
import os


class _JsonFormatter(logging.Formatter):
    """Lightweight JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(*, level: str | None = None, as_json: bool | None = None) -> None:
    """Configure root logger once for scripts and services."""
    env_level = os.getenv("FPL_LOG_LEVEL")
    resolved_level = (level or env_level or "INFO").upper()
    use_json = as_json if as_json is not None else os.getenv("FPL_LOG_JSON", "0") == "1"

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved_level, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Get a logger with a basic stream handler if none exists."""
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
