"""Point-in-time aware join helpers."""

from __future__ import annotations

import pandas as pd


def point_in_time_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    key_cols: list[str],
    left_time_col: str,
    right_effective_col: str,
) -> pd.DataFrame:
    """Join left rows to the latest right record effective at or before left timestamp."""
    left = left_df.copy().sort_values(left_time_col)
    right = right_df.copy().sort_values(right_effective_col)

    joined = pd.merge_asof(
        left,
        right,
        left_on=left_time_col,
        right_on=right_effective_col,
        by=key_cols,
        direction="backward",
        allow_exact_matches=True,
    )
    return joined
