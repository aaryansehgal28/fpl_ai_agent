from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_ai_agent.features.store import build_player_gameweek_features, validate_feature_grain


def test_validate_feature_grain_rejects_duplicates() -> None:
    df = pd.DataFrame(
        [
            {"player_id": "p1", "season": 2024, "gameweek": 1},
            {"player_id": "p1", "season": 2024, "gameweek": 1},
        ]
    )
    with pytest.raises(ValueError):
        validate_feature_grain(df)


def test_build_player_gameweek_features_leakage_safe_form() -> None:
    df = pd.read_csv(Path("tests/fixtures/raw/fpl_players_sample.csv"))
    features = build_player_gameweek_features(df, window_short=3, window_long=5, min_history=3)

    row = features[(features["player_id"] == "p1") & (features["gameweek"] == 4)].iloc[0]
    expected_form = (8 + 6 + 11) / 3
    assert row["form_points_3"] == pytest.approx(expected_form)
    assert 0.0 <= row["injury_risk_proxy"] <= 1.0
    assert "value_signal" in features.columns
