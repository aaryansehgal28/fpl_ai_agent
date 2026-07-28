"""Forecast inference and contract generation utilities."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from fpl_ai_agent.contracts import ForecastOutputContract
from fpl_ai_agent.forecasting.dataset import WindowedDataset


def build_forecast_contracts(
    windowed: WindowedDataset,
    expected_points: np.ndarray,
    uncertainty: np.ndarray,
    *,
    model_version: str,
    horizon: int = 1,
) -> list[ForecastOutputContract]:
    """Create forecast output contracts from model predictions."""
    contracts: list[ForecastOutputContract] = []
    generated_at = datetime.now(timezone.utc)

    for idx, player_id in enumerate(windowed.player_ids):
        contracts.append(
            ForecastOutputContract(
                player_id=player_id,
                season=windowed.seasons[idx],
                gameweek=windowed.gameweeks[idx],
                horizon=horizon,
                expected_points=float(expected_points[idx]),
                uncertainty=max(float(uncertainty[idx]), 0.0),
                availability_probability=_availability_probability(float(uncertainty[idx])),
                model_version=model_version,
                generated_at=generated_at,
            )
        )
    return contracts


def _availability_probability(uncertainty: float) -> float:
    """Convert uncertainty proxy to bounded availability probability."""
    value = 1.0 / (1.0 + max(uncertainty, 0.0))
    return max(0.0, min(1.0, value))
