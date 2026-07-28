"""Optimisation problem primitives and objective settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OptimisationSettings:
    """Config for multi-gameweek objective construction."""

    horizon_weeks: int = 3
    discount_factor: float = 0.98
    transfer_penalty: float = 4.0
    risk_aversion: float = 0.15


def compute_discounted_value(expected_points: list[float], discount_factor: float) -> float:
    """Compute discounted return across horizon points."""
    return sum((discount_factor**idx) * points for idx, points in enumerate(expected_points))
