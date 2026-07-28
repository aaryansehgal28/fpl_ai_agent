"""FPL event-level scoring utilities with position-aware rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PlayerEventProjection:
    """Projected event counts used for expected FPL scoring."""

    position: str
    minutes: float
    goals: float
    assists: float
    clean_sheet_probability: float
    saves: float
    bonus: float
    yellow_cards: float
    red_cards: float
    own_goals: float
    penalties_missed: float
    penalties_saved: float


def expected_fpl_points(event: PlayerEventProjection) -> float:
    """Compute expected FPL points from projected events."""
    pos = event.position.upper()
    if pos not in {"GK", "DEF", "MID", "FWD"}:
        raise ValueError(f"Unsupported position: {event.position}")

    appearance = _appearance_points(event.minutes)
    goal_points = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}[pos] * event.goals
    assist_points = 3.0 * event.assists
    cs_points = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}[pos] * event.clean_sheet_probability
    save_points = (event.saves / 3.0) if pos == "GK" else 0.0
    bonus_points = event.bonus

    negative = (
        event.yellow_cards
        + 3.0 * event.red_cards
        + 2.0 * event.own_goals
        + 2.0 * event.penalties_missed
    )
    penalty_save_points = 5.0 * event.penalties_saved if pos == "GK" else 0.0

    total = appearance + goal_points + assist_points + cs_points + save_points + bonus_points + penalty_save_points - negative
    return float(total)


def _appearance_points(minutes: float) -> float:
    if minutes <= 0:
        return 0.0
    if minutes < 60:
        return 1.0
    return 2.0
