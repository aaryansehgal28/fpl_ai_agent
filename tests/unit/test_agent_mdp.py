from __future__ import annotations

from fpl_ai_agent.agent.mdp import (
    MDPAction,
    MDPState,
    MDPTransitionAssumptions,
    apply_action,
    build_recommendation,
)


def test_apply_action_transfer_penalty_and_next_state() -> None:
    state = MDPState(
        gameweek=10,
        bank=1.0,
        free_transfers=1,
        squad_player_ids={"a", "b", "c"},
        chips_available={"wildcard": True, "free_hit": True, "bench_boost": True},
    )
    action = MDPAction(
        action_type="weekly_plan",
        payload={
            "transfer_out_ids": ["a"],
            "transfer_in_ids": ["x", "y"],
            "chip": "none",
            "expected_points": 60.0,
            "risk_penalty": 2.5,
        },
    )

    result = apply_action(state, action, assumptions=MDPTransitionAssumptions(transfer_penalty=4.0))
    assert result.next_state.gameweek == 11
    assert result.reward == 60.0 - 4.0 - 2.5
    assert result.next_state.free_transfers == 1


def test_build_recommendation_bounds_confidence() -> None:
    action = MDPAction(action_type="weekly_plan", payload={})
    rec = build_recommendation(action=action, expected_value_delta=5.0, risk_delta=-1.0)
    assert 0.0 <= rec.confidence <= 1.0
    assert "weekly_plan" in rec.explanation_text
