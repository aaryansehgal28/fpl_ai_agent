from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_ai_agent.contracts import OptimiserInputContract
from fpl_ai_agent.features.store import enforce_pre_deadline_only
from fpl_ai_agent.forecasting.dataset import build_temporal_windows
from fpl_ai_agent.optimisation.problem import compute_discounted_value


def test_end_to_end_scaffold_flow() -> None:
    df = pd.read_csv(Path("tests/fixtures/raw/fpl_players_sample.csv"))
    filtered = enforce_pre_deadline_only(df, "available_pre_deadline")
    assert filtered["available_pre_deadline"].all()

    feature_cols = ["minutes", "goals", "assists", "ict", "bps", "price", "opp_strength", "home"]
    windows = build_temporal_windows(
        filtered,
        player_col="player_id",
        time_col="gameweek",
        feature_cols=feature_cols,
        target_col="target_points",
        window_length=3,
    )
    assert windows.x.shape[0] > 0

    expected_points_horizon = [float(windows.y.mean()), float(windows.y.mean()), float(windows.y.mean())]
    discounted = compute_discounted_value(expected_points_horizon, discount_factor=0.98)
    assert discounted > 0

    contract = OptimiserInputContract(
        player_id="p1",
        player_name="Player One",
        team="ARS",
        position="MID",
        cost=7.8,
        expected_points_horizon=expected_points_horizon,
        uncertainty_horizon=[1.2, 1.2, 1.2],
        availability_probability=0.95,
        transfer_context={"free_transfers": 1, "bank": 1.2},
    )
    assert contract.expected_points_horizon[0] > 0
