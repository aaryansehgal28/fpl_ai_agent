"""Feature store interfaces at player-season-gameweek grain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureStoreSpec:
    """Metadata for feature storage and expected grain."""

    grain: str = "player-season-gameweek"
    leakage_guard_enabled: bool = True


REQUIRED_GRAIN_COLUMNS = ("player_id", "season", "gameweek")


def enforce_pre_deadline_only(df: pd.DataFrame, availability_col: str) -> pd.DataFrame:
    """Keep only rows with features available before the decision deadline."""
    if availability_col not in df.columns:
        raise KeyError(f"Missing availability column: {availability_col}")
    return df[df[availability_col].astype(bool)].copy()


def validate_feature_grain(df: pd.DataFrame) -> None:
    """Validate player-season-gameweek grain uniqueness."""
    missing = [col for col in REQUIRED_GRAIN_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing grain columns: {missing}")

    duplicate_rows = df.duplicated(subset=list(REQUIRED_GRAIN_COLUMNS)).sum()
    if duplicate_rows:
        raise ValueError(f"Found {duplicate_rows} duplicate rows at player-season-gameweek grain.")


def build_player_gameweek_features(
    df: pd.DataFrame,
    *,
    window_short: int = 3,
    window_long: int = 5,
    min_history: int = 3,
) -> pd.DataFrame:
    """Build leakage-safe rolling features at player-season-gameweek grain."""
    validate_feature_grain(df)
    base = enforce_pre_deadline_only(df, "available_pre_deadline")
    base = base.sort_values(["player_id", "season", "gameweek"]).copy()

    grouped = base.groupby(["player_id", "season"], sort=False)

    # Use shifted values so each gameweek uses only prior information.
    base["minutes_prev"] = grouped["minutes"].shift(1)
    base["goals_prev"] = grouped["goals"].shift(1)
    base["assists_prev"] = grouped["assists"].shift(1)
    base["ict_prev"] = grouped["ict"].shift(1)
    base["bps_prev"] = grouped["bps"].shift(1)
    base["price_prev"] = grouped["price"].shift(1)
    base["opp_strength_prev"] = grouped["opp_strength"].shift(1)
    base["home_prev"] = grouped["home"].shift(1)
    base["points_prev"] = grouped["target_points"].shift(1)

    base["form_points_3"] = grouped["target_points"].transform(
        lambda s: s.shift(1).rolling(window_short, min_periods=1).mean()
    )
    base["form_points_5"] = grouped["target_points"].transform(
        lambda s: s.shift(1).rolling(window_long, min_periods=1).mean()
    )
    base["minutes_rolling_3"] = grouped["minutes"].transform(
        lambda s: s.shift(1).rolling(window_short, min_periods=1).mean()
    )
    base["ict_rolling_3"] = grouped["ict"].transform(
        lambda s: s.shift(1).rolling(window_short, min_periods=1).mean()
    )
    base["bps_rolling_3"] = grouped["bps"].transform(
        lambda s: s.shift(1).rolling(window_short, min_periods=1).mean()
    )

    expected_involvement_raw = base["goals"] * 4.0 + base["assists"] * 3.0
    base["expected_involvement_proxy"] = expected_involvement_raw.groupby(
        [base["player_id"], base["season"]]
    ).transform(lambda s: s.shift(1).rolling(window_short, min_periods=1).mean())

    base["fixture_ease"] = 6.0 - base["opp_strength_prev"].fillna(base["opp_strength"])
    base["fixture_difficulty"] = base["opp_strength_prev"].fillna(base["opp_strength"])

    minutes_ratio = (base["minutes"] / 90.0).clip(lower=0.0, upper=1.0)
    base["injury_risk_proxy"] = 1.0 - minutes_ratio.groupby([base["player_id"], base["season"]]).transform(
        lambda s: s.shift(1).rolling(window_short, min_periods=1).mean()
    )
    base["injury_risk_proxy"] = base["injury_risk_proxy"].fillna(0.5).clip(lower=0.0, upper=1.0)

    safe_price = base["price_prev"].replace(0, np.nan)
    base["value_signal"] = (base["form_points_3"] / safe_price).replace([np.inf, -np.inf], np.nan)

    base["history_count"] = grouped.cumcount()
    features = base[base["history_count"] >= min_history].copy()
    features = features.dropna(
        subset=[
            "form_points_3",
            "form_points_5",
            "minutes_rolling_3",
            "ict_rolling_3",
            "bps_rolling_3",
            "expected_involvement_proxy",
            "value_signal",
        ]
    )
    return features.reset_index(drop=True)
