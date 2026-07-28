"""Starting XI, bench order, and captaincy optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fpl_ai_agent.contracts import OptimiserInputContract


@dataclass(slots=True)
class LineupSettings:
    """Settings for one-gameweek lineup decisions."""

    horizon_index: int = 0
    vice_activation_weight: float = 0.15


@dataclass(slots=True)
class LineupPlanResult:
    """Optimized weekly lineup and captaincy plan."""

    starter_ids: list[str]
    bench_order_ids: list[str]
    captain_id: str
    vice_captain_id: str
    expected_lineup_points: float


def optimize_lineup(
    squad_players: Iterable[OptimiserInputContract],
    *,
    settings: LineupSettings,
    chip_used: str = "none",
) -> LineupPlanResult:
    """Optimize starting XI, bench order, captain, and vice-captain."""
    try:
        import pulp
    except Exception as exc:  # pragma: no cover
        raise ImportError("PuLP is required for lineup optimization. Install pulp in the active env.") from exc

    players = list(squad_players)
    if len(players) != 15:
        raise ValueError("Lineup optimization requires exactly 15 selected squad players.")

    idx = list(range(len(players)))
    prob = pulp.LpProblem("fpl_weekly_lineup", pulp.LpMaximize)

    start = pulp.LpVariable.dicts("start", idx, lowBound=0, upBound=1, cat="Binary")
    bench = pulp.LpVariable.dicts("bench", idx, lowBound=0, upBound=1, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", idx, lowBound=0, upBound=1, cat="Binary")
    vice = pulp.LpVariable.dicts("vice", idx, lowBound=0, upBound=1, cat="Binary")
    bench_slot = pulp.LpVariable.dicts("bench_slot", [(i, s) for i in idx for s in range(1, 5)], lowBound=0, upBound=1, cat="Binary")

    # Starting XI and bench constraints.
    prob += pulp.lpSum(start[i] for i in idx) == 11
    prob += pulp.lpSum(bench[i] for i in idx) == 4
    for i in idx:
        prob += start[i] + bench[i] == 1

    # Formation constraints for starters.
    prob += pulp.lpSum(start[i] for i in idx if players[i].position == "GK") == 1
    prob += pulp.lpSum(start[i] for i in idx if players[i].position == "DEF") >= 3
    prob += pulp.lpSum(start[i] for i in idx if players[i].position == "MID") >= 2
    prob += pulp.lpSum(start[i] for i in idx if players[i].position == "FWD") >= 1

    # Captain and vice-captain constraints.
    prob += pulp.lpSum(captain[i] for i in idx) == 1
    prob += pulp.lpSum(vice[i] for i in idx) == 1
    for i in idx:
        prob += captain[i] <= start[i]
        prob += vice[i] <= start[i]
        prob += captain[i] + vice[i] <= 1

    # Bench order constraints.
    for i in idx:
        prob += pulp.lpSum(bench_slot[(i, s)] for s in range(1, 5)) == bench[i]
    for s in range(1, 5):
        prob += pulp.lpSum(bench_slot[(i, s)] for i in idx) == 1

    # Keep exactly one GK on bench due to total squad composition.
    prob += pulp.lpSum(bench[i] for i in idx if players[i].position == "GK") == 1

    expected_points = []
    for i in idx:
        horizon = players[i].expected_points_horizon
        uncertainty = players[i].uncertainty_horizon
        h = min(settings.horizon_index, len(horizon) - 1)
        expected = horizon[h]
        avail = players[i].availability_probability
        expected_points.append((expected, max(0.0, avail), uncertainty[h] if h < len(uncertainty) else 0.0))

    starter_points = pulp.lpSum(start[i] * expected_points[i][0] * expected_points[i][1] for i in idx)
    captain_bonus = pulp.lpSum(captain[i] * expected_points[i][0] * expected_points[i][1] for i in idx)
    vice_fallback = pulp.lpSum(
        vice[i] * expected_points[i][0] * (1.0 - expected_points[i][1]) * settings.vice_activation_weight for i in idx
    )

    bench_points = pulp.lpSum(bench[i] * expected_points[i][0] * expected_points[i][1] for i in idx)
    if chip_used == "bench_boost":
        objective = starter_points + captain_bonus + vice_fallback + bench_points
    else:
        objective = starter_points + captain_bonus + vice_fallback

    prob += objective
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Lineup optimization did not find optimal solution: {pulp.LpStatus[prob.status]}")

    starter_ids = [players[i].player_id for i in idx if float(start[i].value() or 0.0) > 0.5]

    bench_by_slot: dict[int, str] = {}
    for s in range(1, 5):
        for i in idx:
            if float(bench_slot[(i, s)].value() or 0.0) > 0.5:
                bench_by_slot[s] = players[i].player_id
                break
    bench_order_ids = [bench_by_slot[s] for s in range(1, 5)]

    captain_id = next(players[i].player_id for i in idx if float(captain[i].value() or 0.0) > 0.5)
    vice_id = next(players[i].player_id for i in idx if float(vice[i].value() or 0.0) > 0.5)

    return LineupPlanResult(
        starter_ids=starter_ids,
        bench_order_ids=bench_order_ids,
        captain_id=captain_id,
        vice_captain_id=vice_id,
        expected_lineup_points=float(pulp.value(objective)),
    )
