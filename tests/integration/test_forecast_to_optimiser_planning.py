from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_ai_agent.contracts import ForecastOutputContract
from fpl_ai_agent.optimisation.candidate_builder import build_optimiser_candidates
from fpl_ai_agent.optimisation.problem import OptimisationSettings, TransferContext, optimize_squad


def test_forecast_to_optimiser_flow() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    forecast_rows = json.loads(Path("tests/fixtures/forecast_contracts_sample.json").read_text(encoding="utf-8"))
    forecasts = [ForecastOutputContract(**row) for row in forecast_rows]

    metadata = pd.DataFrame(json.loads(Path("tests/fixtures/optimiser_candidates.json").read_text(encoding="utf-8")))
    metadata = metadata[["player_id", "player_name", "team", "position", "cost"]]

    candidates = build_optimiser_candidates(forecasts, player_metadata=metadata, horizon_weeks=3)

    # Add enough extra candidates to satisfy legal squad constraints.
    full_pool = [
        *json.loads(Path("tests/fixtures/optimiser_candidates.json").read_text(encoding="utf-8")),
    ]
    from fpl_ai_agent.contracts import OptimiserInputContract

    seed = [OptimiserInputContract(**row) for row in full_pool]
    seed_by_id = {c.player_id: c for c in seed}
    merged: list[OptimiserInputContract] = []
    seen = set()
    for c in candidates + seed:
        if c.player_id not in seen:
            merged.append(c if c.player_id not in seed_by_id else seed_by_id[c.player_id])
            seen.add(c.player_id)

    result = optimize_squad(
        merged,
        settings=OptimisationSettings(budget=100.0),
        transfer_context=TransferContext(current_squad_ids=set(), free_transfers=1),
    )
    assert len(result.selected_player_ids) == 15
