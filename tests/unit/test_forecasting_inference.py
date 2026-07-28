from __future__ import annotations

import numpy as np

from fpl_ai_agent.forecasting.dataset import WindowedDataset
from fpl_ai_agent.forecasting.inference import build_forecast_contracts


def test_build_forecast_contracts() -> None:
    windowed = WindowedDataset(
        x=np.zeros((2, 3, 4), dtype=float),
        y=np.array([5.0, 6.0], dtype=float),
        player_ids=["p1", "p2"],
        seasons=[2024, 2024],
        gameweeks=[6, 6],
    )
    expected = np.array([5.5, 6.2], dtype=float)
    uncertainty = np.array([1.0, 0.5], dtype=float)

    contracts = build_forecast_contracts(
        windowed,
        expected,
        uncertainty,
        model_version="temporal_cnn_v1",
        horizon=1,
    )
    assert len(contracts) == 2
    assert contracts[0].expected_points == 5.5
    assert 0.0 <= contracts[0].availability_probability <= 1.0
