from __future__ import annotations

from pathlib import Path

import pytest


def test_run_backtest_script_runs(tmp_path) -> None:
    pytest.importorskip("pulp", reason="pulp required")

    from scripts.run_backtest import main
    import sys

    output_path = tmp_path / "report.json"
    argv = [
        "run_backtest.py",
        "--config",
        "configs/base.yaml",
        "--raw",
        "tests/fixtures/raw/fpl_players_sample.csv",
        "--candidates",
        "tests/fixtures/optimiser_candidates.json",
        "--output",
        str(output_path),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old_argv

    assert output_path.exists()
    assert output_path.stat().st_size > 0
