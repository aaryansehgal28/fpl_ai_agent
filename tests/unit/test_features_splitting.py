from __future__ import annotations

import pandas as pd

from fpl_ai_agent.features.splitting import fit_standardizer, split_and_standardize


def _multi_season_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2023, 2023, 2024, 2024],
            "signal": [1.0, 2.0, 10.0, 12.0, 20.0, 24.0],
            "other": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
        }
    )


def test_fit_standardizer_uses_train_only() -> None:
    df = _multi_season_df()
    train = df[df["season"] == 2022]
    scaler = fit_standardizer(train, ["signal"])
    assert scaler.means["signal"] == 1.5


def test_split_and_standardize_chronological() -> None:
    df = _multi_season_df()
    split = split_and_standardize(
        df,
        season_col="season",
        train_seasons={2022},
        valid_seasons={2023},
        test_seasons={2024},
        feature_cols=["signal", "other"],
    )

    assert set(split.train["season"].unique()) == {2022}
    assert set(split.valid["season"].unique()) == {2023}
    assert set(split.test["season"].unique()) == {2024}

    # Train split should be centered after standardization.
    assert round(float(split.train["signal"].mean()), 10) == 0.0
