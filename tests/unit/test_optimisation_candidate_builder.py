from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fpl_ai_agent.contracts import ForecastOutputContract
from fpl_ai_agent.optimisation.candidate_builder import build_optimiser_candidates


def test_build_optimiser_candidates_from_forecasts() -> None:
    forecast_rows = json.loads(Path("tests/fixtures/forecast_contracts_sample.json").read_text(encoding="utf-8"))
    forecasts = [ForecastOutputContract(**row) for row in forecast_rows]
    metadata_rows = json.loads(Path("tests/fixtures/optimiser_candidates.json").read_text(encoding="utf-8"))
    metadata = pd.DataFrame(metadata_rows)[["player_id", "player_name", "team", "position", "cost"]]

    candidates = build_optimiser_candidates(forecasts, player_metadata=metadata, horizon_weeks=3)
    assert len(candidates) == 2
    assert candidates[0].expected_points_horizon and len(candidates[0].expected_points_horizon) == 3
    assert candidates[0].uncertainty_horizon and len(candidates[0].uncertainty_horizon) == 3
