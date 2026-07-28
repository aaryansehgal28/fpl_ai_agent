from __future__ import annotations

from fpl_ai_agent.optimisation.problem import compute_discounted_value


def test_compute_discounted_value() -> None:
    points = [10.0, 8.0, 6.0]
    value = compute_discounted_value(points, discount_factor=0.9)
    assert round(value, 4) == round(10.0 + 0.9 * 8.0 + 0.81 * 6.0, 4)
