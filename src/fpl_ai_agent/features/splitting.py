"""Leakage-safe chronological splits and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ChronologicalSplit:
    """Container for train/validation/test splits."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass(slots=True)
class FeatureStandardizer:
    """Per-feature mean/std fitted only on training data."""

    means: dict[str, float]
    stds: dict[str, float]


def split_by_season_chronological(
    df: pd.DataFrame,
    *,
    season_col: str,
    train_seasons: set[int],
    valid_seasons: set[int],
    test_seasons: set[int],
) -> ChronologicalSplit:
    """Split data by non-overlapping season sets with chronological guardrails."""
    if train_seasons & valid_seasons or train_seasons & test_seasons or valid_seasons & test_seasons:
        raise ValueError("train/valid/test season sets must be disjoint.")

    train = df[df[season_col].isin(train_seasons)].copy()
    valid = df[df[season_col].isin(valid_seasons)].copy()
    test = df[df[season_col].isin(test_seasons)].copy()

    if not train.empty and not valid.empty and train[season_col].max() >= valid[season_col].min():
        raise ValueError("Train seasons must be strictly earlier than validation seasons.")
    if not valid.empty and not test.empty and valid[season_col].max() >= test[season_col].min():
        raise ValueError("Validation seasons must be strictly earlier than test seasons.")

    return ChronologicalSplit(train=train, valid=valid, test=test)


def fit_standardizer(train_df: pd.DataFrame, feature_cols: list[str]) -> FeatureStandardizer:
    """Fit mean/std on training subset only."""
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in feature_cols:
        means[col] = float(train_df[col].mean())
        std = float(train_df[col].std(ddof=0))
        stds[col] = std if std > 0 else 1.0
    return FeatureStandardizer(means=means, stds=stds)


def apply_standardizer(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    standardizer: FeatureStandardizer,
) -> pd.DataFrame:
    """Apply pre-fitted train-only standardizer to any split."""
    transformed = df.copy()
    for col in feature_cols:
        transformed[col] = (transformed[col] - standardizer.means[col]) / standardizer.stds[col]
    return transformed


def split_and_standardize(
    df: pd.DataFrame,
    *,
    season_col: str,
    train_seasons: set[int],
    valid_seasons: set[int],
    test_seasons: set[int],
    feature_cols: list[str],
) -> ChronologicalSplit:
    """Create chronological splits and standardize with training-fit parameters."""
    split = split_by_season_chronological(
        df,
        season_col=season_col,
        train_seasons=train_seasons,
        valid_seasons=valid_seasons,
        test_seasons=test_seasons,
    )
    standardizer = fit_standardizer(split.train, feature_cols)
    return ChronologicalSplit(
        train=apply_standardizer(split.train, feature_cols=feature_cols, standardizer=standardizer),
        valid=apply_standardizer(split.valid, feature_cols=feature_cols, standardizer=standardizer),
        test=apply_standardizer(split.test, feature_cols=feature_cols, standardizer=standardizer),
    )
