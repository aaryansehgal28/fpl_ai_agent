"""Chronological split helpers for leak-safe training and evaluation."""

from __future__ import annotations

import pandas as pd


def split_by_season(
    df: pd.DataFrame,
    season_col: str,
    train_seasons: set[int],
    valid_seasons: set[int],
    test_seasons: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by explicit season sets."""
    train = df[df[season_col].isin(train_seasons)].copy()
    valid = df[df[season_col].isin(valid_seasons)].copy()
    test = df[df[season_col].isin(test_seasons)].copy()
    return train, valid, test
