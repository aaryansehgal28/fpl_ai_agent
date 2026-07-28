"""MDP scaffolding for weekly FPL decision recommendation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MDPState:
    """Compact representation of team context at a gameweek."""

    gameweek: int
    bank: float
    free_transfers: int


@dataclass(slots=True)
class MDPAction:
    """Transfer or chip action candidate."""

    action_type: str
    payload: dict[str, str | int | float]


@dataclass(slots=True)
class Recommendation:
    """Human-approval recommendation output."""

    action: MDPAction
    confidence: float
    rationale: str
