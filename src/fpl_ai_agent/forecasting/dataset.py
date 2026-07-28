"""Temporal dataset builders for CNN-ready windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class WindowedDataset:
    """Container for (samples, window_length, features) tensors and targets."""

    x: np.ndarray
    y: np.ndarray
    player_ids: list[str]
    seasons: list[int]
    gameweeks: list[int]


def build_temporal_windows(
    df: pd.DataFrame,
    *,
    player_col: str,
    time_col: str,
    feature_cols: list[str],
    target_col: str,
    window_length: int,
) -> WindowedDataset:
    """Build per-player rolling windows with strict temporal ordering."""
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    player_ids: list[str] = []
    seasons: list[int] = []
    gameweeks: list[int] = []

    ordered = df.sort_values([player_col, time_col]).reset_index(drop=True)
    for player_id, grp in ordered.groupby(player_col, sort=False):
        grp = grp.reset_index(drop=True)
        if len(grp) <= window_length:
            continue
        values = grp[feature_cols].to_numpy(dtype=float)
        targets = grp[target_col].to_numpy(dtype=float)
        for idx in range(window_length, len(grp)):
            rows_x.append(values[idx - window_length : idx])
            rows_y.append(targets[idx])
            player_ids.append(str(player_id))
            seasons.append(int(grp.loc[idx, "season"]))
            gameweeks.append(int(grp.loc[idx, "gameweek"]))

    if not rows_x:
        x = np.empty((0, window_length, len(feature_cols)), dtype=float)
        y = np.empty((0,), dtype=float)
    else:
        x = np.stack(rows_x)
        y = np.asarray(rows_y, dtype=float)

    return WindowedDataset(
        x=x,
        y=y,
        player_ids=player_ids,
        seasons=seasons,
        gameweeks=gameweeks,
    )
