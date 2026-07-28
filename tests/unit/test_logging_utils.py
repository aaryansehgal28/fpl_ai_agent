from __future__ import annotations

import logging

from fpl_ai_agent.utils.logging_utils import configure_logging, get_logger


def test_configure_logging_json_sets_handler(monkeypatch) -> None:
    monkeypatch.setenv("FPL_LOG_JSON", "1")
    monkeypatch.setenv("FPL_LOG_LEVEL", "DEBUG")
    configure_logging()

    root = logging.getLogger()
    assert root.handlers
    assert root.level == logging.DEBUG


def test_get_logger_uses_root_configuration() -> None:
    configure_logging(level="INFO", as_json=False)
    logger = get_logger("fpl.test")
    assert logger.name == "fpl.test"
