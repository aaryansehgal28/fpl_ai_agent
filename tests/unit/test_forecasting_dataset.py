from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_ai_agent.forecasting.dataset import build_temporal_windows


def test_build_temporal_windows_shape() -> None:
    df = pd.read_csv(Path("tests/fixtures/raw/fpl_players_sample.csv"))
    feature_cols = [
        "minutes",
        "goals",
        "assists",
        "ict",
        "bps",
        "price",
        "opp_strength",
        "home",
    ]
    result = build_temporal_windows(
        df,
        player_col="player_id",
        time_col="gameweek",
        feature_cols=feature_cols,
        target_col="target_points",
        window_length=3,
    )
    assert result.x.ndim == 3
    assert result.x.shape[1] == 3
    assert result.x.shape[2] == len(feature_cols)
    assert result.y.shape[0] == result.x.shape[0]
