"""Feature store interfaces at player-season-gameweek grain."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class FeatureStoreSpec:
    """Metadata for feature storage and expected grain."""

    grain: str = "player-season-gameweek"
    leakage_guard_enabled: bool = True


def enforce_pre_deadline_only(df: pd.DataFrame, availability_col: str) -> pd.DataFrame:
    """Keep only rows with features available before the decision deadline."""
    if availability_col not in df.columns:
        raise KeyError(f"Missing availability column: {availability_col}")
    return df[df[availability_col].astype(bool)].copy()
