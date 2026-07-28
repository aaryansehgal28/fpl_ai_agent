from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_ai_agent.features.splitting import split_and_standardize
from fpl_ai_agent.features.store import build_player_gameweek_features
from fpl_ai_agent.forecasting.dataset import build_temporal_windows


def test_feature_pipeline_to_temporal_windows() -> None:
    df = pd.read_csv(Path("tests/fixtures/raw/fpl_players_sample.csv"))
    features = build_player_gameweek_features(df, min_history=3)
    features = features.copy()

    feature_cols = [
        "minutes_prev",
        "goals_prev",
        "assists_prev",
        "ict_prev",
        "bps_prev",
        "price_prev",
        "fixture_difficulty",
        "home_prev",
        "form_points_3",
        "expected_involvement_proxy",
        "injury_risk_proxy",
        "value_signal",
    ]

    standardized = split_and_standardize(
        features,
        season_col="season",
        train_seasons={2024},
        valid_seasons=set(),
        test_seasons=set(),
        feature_cols=feature_cols,
    )

    windows = build_temporal_windows(
        standardized.train,
        player_col="player_id",
        time_col="gameweek",
        feature_cols=feature_cols,
        target_col="target_points",
        window_length=2,
    )

    assert windows.x.shape[0] > 0
    assert windows.x.shape[1] == 2
    assert windows.x.shape[2] == len(feature_cols)
