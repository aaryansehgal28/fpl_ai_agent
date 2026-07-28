from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpl_ai_agent.contracts import OptimiserInputContract
from fpl_ai_agent.optimisation.lineup import LineupSettings, optimize_lineup
from fpl_ai_agent.optimisation.problem import OptimisationSettings, TransferContext, optimize_squad


def _load_squad() -> list[OptimiserInputContract]:
    rows = json.loads(Path("tests/fixtures/optimiser_candidates.json").read_text(encoding="utf-8"))
    players = [OptimiserInputContract(**row) for row in rows]
    squad = optimize_squad(
        players,
        settings=OptimisationSettings(budget=100.0),
        transfer_context=TransferContext(current_squad_ids=set(), free_transfers=1),
    )
    selected = [player for player in players if player.player_id in set(squad.selected_player_ids)]
    return selected


def test_optimize_lineup_constraints() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    squad = _load_squad()
    result = optimize_lineup(squad, settings=LineupSettings(horizon_index=0), chip_used="none")

    assert len(result.starter_ids) == 11
    assert len(result.bench_order_ids) == 4
    assert result.captain_id in result.starter_ids
    assert result.vice_captain_id in result.starter_ids
    assert result.captain_id != result.vice_captain_id


def test_bench_boost_increases_objective() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    squad = _load_squad()
    no_chip = optimize_lineup(squad, settings=LineupSettings(horizon_index=0), chip_used="none")
    with_chip = optimize_lineup(squad, settings=LineupSettings(horizon_index=0), chip_used="bench_boost")
    assert with_chip.expected_lineup_points >= no_chip.expected_lineup_points
