"""Build optimizer candidates from forecast outputs and player metadata."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from fpl_ai_agent.contracts import ForecastOutputContract, OptimiserInputContract


def build_optimiser_candidates(
    forecasts: list[ForecastOutputContract],
    *,
    player_metadata: pd.DataFrame,
    horizon_weeks: int,
) -> list[OptimiserInputContract]:
    """Construct optimizer candidate contracts from forecast rows."""
    grouped: dict[str, list[ForecastOutputContract]] = defaultdict(list)
    for forecast in forecasts:
        grouped[forecast.player_id].append(forecast)

    meta_idx = player_metadata.drop_duplicates(subset=["player_id"]).set_index("player_id")
    candidates: list[OptimiserInputContract] = []

    for player_id, rows in grouped.items():
        if player_id not in meta_idx.index:
            continue
        meta = meta_idx.loc[player_id]
        ordered = sorted(rows, key=lambda row: row.horizon)

        exp = [float(row.expected_points) for row in ordered]
        unc = [float(row.uncertainty) for row in ordered]
        availability = min(float(row.availability_probability) for row in ordered)

        exp = _pad_to_horizon(exp, horizon_weeks)
        unc = _pad_to_horizon(unc, horizon_weeks)

        cost = float(meta["cost"] if "cost" in meta else meta.get("price", 0.0))
        candidates.append(
            OptimiserInputContract(
                player_id=player_id,
                player_name=str(meta.get("player_name", player_id)),
                team=str(meta["team"]),
                position=str(meta["position"]),
                cost=cost,
                expected_points_horizon=exp,
                uncertainty_horizon=unc,
                availability_probability=availability,
                transfer_context={},
            )
        )

    return candidates


def _pad_to_horizon(values: list[float], horizon: int) -> list[float]:
    """Pad series to required horizon by repeating the last observation."""
    if not values:
        return [0.0 for _ in range(horizon)]
    if len(values) >= horizon:
        return values[:horizon]
    return values + [values[-1] for _ in range(horizon - len(values))]
