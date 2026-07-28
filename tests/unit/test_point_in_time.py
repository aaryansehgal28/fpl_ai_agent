from __future__ import annotations

import pandas as pd

from fpl_ai_agent.identity.point_in_time import point_in_time_join


def test_point_in_time_join_uses_latest_effective_record() -> None:
    left = pd.DataFrame(
        {
            "team": ["ARS", "ARS"],
            "match_date": pd.to_datetime(["2025-01-10", "2025-01-20"]),
        }
    )
    right = pd.DataFrame(
        {
            "team": ["ARS", "ARS"],
            "effective_date": pd.to_datetime(["2025-01-01", "2025-01-15"]),
            "elo": [1800, 1825],
        }
    )

    joined = point_in_time_join(
        left,
        right,
        key_cols=["team"],
        left_time_col="match_date",
        right_effective_col="effective_date",
    )

    assert joined.loc[0, "elo"] == 1800
    assert joined.loc[1, "elo"] == 1825
