"""Logging helpers for consistent module-level loggers."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with a basic stream handler if none exists."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
