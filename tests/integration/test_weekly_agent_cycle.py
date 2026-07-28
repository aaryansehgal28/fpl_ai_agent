from __future__ import annotations

import pytest


def test_run_agent_weekly_script_runs() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    from scripts.run_agent_weekly import main
    import sys

    argv = [
        "run_agent_weekly.py",
        "--config",
        "configs/base.yaml",
        "--candidates",
        "tests/fixtures/optimiser_candidates.json",
        "--gameweek",
        "10",
        "--free-transfers",
        "1",
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old_argv
