from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_ai_agent.backtesting.engine import run_walk_forward_backtest
from fpl_ai_agent.contracts import OptimiserInputContract
from fpl_ai_agent.optimisation.problem import OptimisationSettings


def test_run_walk_forward_backtest_smoke() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    raw = pd.read_csv(Path("tests/fixtures/raw/fpl_players_sample.csv"))
    rows = json.loads(Path("tests/fixtures/optimiser_candidates.json").read_text(encoding="utf-8"))
    candidates = [OptimiserInputContract(**row) for row in rows]

    result = run_walk_forward_backtest(
        raw_df=raw,
        candidate_pool=candidates,
        optimisation_settings=OptimisationSettings(budget=100.0),
        discount_factor=0.98,
        transfer_penalty=4.0,
    )

    assert "mae" in result.forecast_metrics
    assert result.decision_metrics.ci_lower <= result.decision_metrics.ci_upper
    assert len(result.weeks) > 0
