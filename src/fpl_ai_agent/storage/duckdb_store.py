"""DuckDB storage helpers."""

from __future__ import annotations

from pathlib import Path


def connect(path: str | Path):
    """Open a DuckDB connection, creating parent directories if needed."""
    import duckdb

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))
