from __future__ import annotations

import pandas as pd

from fpl_ai_agent.backtesting.walk_forward import split_by_season


def test_split_by_season() -> None:
    df = pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2025],
            "value": [1, 2, 3, 4],
        }
    )
    train, valid, test = split_by_season(
        df,
        season_col="season",
        train_seasons={2022, 2023},
        valid_seasons={2024},
        test_seasons={2025},
    )
    assert train.shape[0] == 2
    assert valid.shape[0] == 1
    assert test.shape[0] == 1
