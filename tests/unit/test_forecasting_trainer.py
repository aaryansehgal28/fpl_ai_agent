from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_ai_agent.features.store import build_player_gameweek_features
from fpl_ai_agent.forecasting.trainer import ForecastTrainerConfig, predict_with_uncertainty, train_forecaster


def test_train_forecaster_and_predict_with_uncertainty() -> None:
    pytest.importorskip("torch")

    raw = pd.read_csv(Path("tests/fixtures/raw/fpl_players_sample.csv"))
    feat = build_player_gameweek_features(raw, min_history=3)

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

    cfg = ForecastTrainerConfig(
        feature_cols=feature_cols,
        target_col="target_points",
        window_length=2,
        train_seasons={2024},
        valid_seasons=set(),
        test_seasons=set(),
        epochs=3,
        batch_size=8,
        learning_rate=1e-3,
        model_version="test_model",
    )

    bundle = train_forecaster(feat, cfg)
    windowed, means, stds = predict_with_uncertainty(bundle, feat, target_col="target_points")

    assert means.shape[0] == stds.shape[0]
    assert means.shape[0] == len(windowed.player_ids)
    assert bundle.metrics["valid_mae"] >= 0.0
