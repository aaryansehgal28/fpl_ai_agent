from __future__ import annotations

from fpl_ai_agent.evaluation.fpl_scoring import PlayerEventProjection, expected_fpl_points


def test_position_weighted_goal_scoring() -> None:
    defender = PlayerEventProjection(
        position="DEF",
        minutes=90,
        goals=1.0,
        assists=0.0,
        clean_sheet_probability=1.0,
        saves=0.0,
        bonus=0.0,
        yellow_cards=0.0,
        red_cards=0.0,
        own_goals=0.0,
        penalties_missed=0.0,
        penalties_saved=0.0,
    )
    forward = PlayerEventProjection(
        position="FWD",
        minutes=90,
        goals=1.0,
        assists=0.0,
        clean_sheet_probability=1.0,
        saves=0.0,
        bonus=0.0,
        yellow_cards=0.0,
        red_cards=0.0,
        own_goals=0.0,
        penalties_missed=0.0,
        penalties_saved=0.0,
    )
    assert expected_fpl_points(defender) > expected_fpl_points(forward)


def test_cards_and_own_goal_penalties() -> None:
    player = PlayerEventProjection(
        position="MID",
        minutes=90,
        goals=0.0,
        assists=0.0,
        clean_sheet_probability=0.0,
        saves=0.0,
        bonus=0.0,
        yellow_cards=1.0,
        red_cards=0.0,
        own_goals=1.0,
        penalties_missed=0.0,
        penalties_saved=0.0,
    )
    assert expected_fpl_points(player) < 2.0
