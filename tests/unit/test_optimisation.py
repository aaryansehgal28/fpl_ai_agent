from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpl_ai_agent.contracts import OptimiserInputContract
from fpl_ai_agent.optimisation.problem import compute_discounted_value
from fpl_ai_agent.optimisation.problem import (
    OptimisationSettings,
    TransferContext,
    compute_discounted_risk,
    optimize_squad,
)


def test_compute_discounted_value() -> None:
    points = [10.0, 8.0, 6.0]
    value = compute_discounted_value(points, discount_factor=0.9)
    assert round(value, 4) == round(10.0 + 0.9 * 8.0 + 0.81 * 6.0, 4)


def test_compute_discounted_risk() -> None:
    unc = [2.0, 1.0, 0.5]
    risk = compute_discounted_risk(unc, discount_factor=0.9)
    assert round(risk, 4) == round(2.0 + 0.9 * 1.0 + 0.81 * 0.5, 4)


def _load_candidates() -> list[OptimiserInputContract]:
    rows = json.loads(Path("tests/fixtures/optimiser_candidates.json").read_text(encoding="utf-8"))
    return [OptimiserInputContract(**row) for row in rows]


def test_optimize_squad_respects_legal_constraints() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    candidates = _load_candidates()
    settings = OptimisationSettings(
        horizon_weeks=3,
        discount_factor=0.98,
        transfer_penalty=4.0,
        risk_aversion=0.15,
        budget=100.0,
    )
    context = TransferContext(current_squad_ids=set(), free_transfers=1)
    result = optimize_squad(candidates, settings=settings, transfer_context=context)

    selected = {player.player_id: player for player in candidates if player.player_id in result.selected_player_ids}
    assert len(selected) == 15
    assert sum(player.position == "GK" for player in selected.values()) == 2
    assert sum(player.position == "DEF" for player in selected.values()) == 5
    assert sum(player.position == "MID" for player in selected.values()) == 5
    assert sum(player.position == "FWD" for player in selected.values()) == 3
    assert sum(player.cost for player in selected.values()) <= 100.0 + 1e-6
    assert result.next_free_transfers in {1, 2}


def test_free_transfer_reduces_paid_transfers() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    candidates = _load_candidates()
    # Start with a current squad that excludes one high-value player likely to be brought in.
    current = {
        "p_gk1", "p_gk2", "p_def1", "p_def2", "p_def3", "p_def4", "p_def5",
        "p_mid1", "p_mid2", "p_mid4", "p_mid5", "p_mid6", "p_fwd1", "p_fwd2", "p_fwd4",
    }
    settings = OptimisationSettings(budget=100.0)
    zero_ft = optimize_squad(
        candidates,
        settings=settings,
        transfer_context=TransferContext(current_squad_ids=current, free_transfers=0),
    )
    one_ft = optimize_squad(
        candidates,
        settings=settings,
        transfer_context=TransferContext(current_squad_ids=current, free_transfers=1),
    )
    assert one_ft.paid_transfers <= zero_ft.paid_transfers
    assert one_ft.transfers_made >= one_ft.paid_transfers


def test_chip_bonus_support() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    candidates = _load_candidates()
    settings = OptimisationSettings(budget=100.0)
    all_ids = {candidate.player_id for candidate in candidates}
    result = optimize_squad(
        candidates,
        settings=settings,
        transfer_context=TransferContext(
            current_squad_ids=all_ids,
            free_transfers=1,
            bench_boost_available=True,
            wildcard_available=True,
            free_hit_available=True,
            bench_boost_bonus=8.0,
            wildcard_bonus=1.0,
            free_hit_bonus=2.0,
        ),
    )
    assert result.chip_used == "bench_boost"


def test_position_risk_weight_changes_defender_choice() -> None:
    pytest.importorskip("pulp", reason="pulp required")

    players = [
        OptimiserInputContract(
            player_id="gk_safe",
            player_name="GK Safe",
            team="ARS",
            position="GK",
            cost=5.0,
            expected_points_horizon=[5.0, 5.0, 5.0],
            uncertainty_horizon=[0.6, 0.6, 0.6],
            availability_probability=1.0,
            transfer_context={},
        ),
        OptimiserInputContract(
            player_id="def_risky",
            player_name="DEF Risky",
            team="LIV",
            position="DEF",
            cost=5.0,
            expected_points_horizon=[7.0, 7.0, 7.0],
            uncertainty_horizon=[2.4, 2.4, 2.4],
            availability_probability=1.0,
            transfer_context={},
        ),
        OptimiserInputContract(
            player_id="def_safe",
            player_name="DEF Safe",
            team="MCI",
            position="DEF",
            cost=5.0,
            expected_points_horizon=[6.2, 6.2, 6.2],
            uncertainty_horizon=[0.6, 0.6, 0.6],
            availability_probability=1.0,
            transfer_context={},
        ),
        OptimiserInputContract(
            player_id="mid_safe",
            player_name="MID Safe",
            team="CHE",
            position="MID",
            cost=5.0,
            expected_points_horizon=[5.8, 5.8, 5.8],
            uncertainty_horizon=[0.6, 0.6, 0.6],
            availability_probability=1.0,
            transfer_context={},
        ),
        OptimiserInputContract(
            player_id="fwd_safe",
            player_name="FWD Safe",
            team="TOT",
            position="FWD",
            cost=5.0,
            expected_points_horizon=[5.8, 5.8, 5.8],
            uncertainty_horizon=[0.6, 0.6, 0.6],
            availability_probability=1.0,
            transfer_context={},
        ),
    ]

    common_kwargs = {
        "horizon_weeks": 3,
        "budget": 30.0,
        "squad_size": 4,
        "max_from_team": 3,
        "position_quota": {"GK": 1, "DEF": 1, "MID": 1, "FWD": 1},
    }
    low_def_penalty = optimize_squad(
        players,
        settings=OptimisationSettings(
            **common_kwargs,
            risk_aversion=0.3,
            position_risk_weights={"GK": 1.0, "DEF": 0.5, "MID": 1.0, "FWD": 1.0},
        ),
        transfer_context=TransferContext(current_squad_ids=set(), free_transfers=1),
    )
    high_def_penalty = optimize_squad(
        players,
        settings=OptimisationSettings(
            **common_kwargs,
            risk_aversion=0.3,
            position_risk_weights={"GK": 1.0, "DEF": 2.0, "MID": 1.0, "FWD": 1.0},
        ),
        transfer_context=TransferContext(current_squad_ids=set(), free_transfers=1),
    )

    assert "def_risky" in set(low_def_penalty.selected_player_ids)
    assert "def_safe" in set(high_def_penalty.selected_player_ids)
