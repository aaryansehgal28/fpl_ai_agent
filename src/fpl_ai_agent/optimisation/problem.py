"""Optimisation problem primitives and objective settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fpl_ai_agent.contracts import OptimiserInputContract


@dataclass(slots=True)
class OptimisationSettings:
    """Config for multi-gameweek objective construction."""

    horizon_weeks: int = 3
    discount_factor: float = 0.98
    transfer_penalty: float = 4.0
    risk_aversion: float = 0.15
    budget: float = 100.0
    squad_size: int = 15
    max_from_team: int = 3
    position_quota: dict[str, int] | None = None
    position_score_weights: dict[str, float] | None = None
    position_risk_weights: dict[str, float] | None = None
    value_signal_weight: float = 0.0
    fixture_difficulty_weight: float = 0.0
    injury_risk_weight: float = 0.0
    rotation_risk_weight: float = 0.0

    def normalized_position_quota(self) -> dict[str, int]:
        """Return FPL position quota defaults when not explicitly set."""
        if self.position_quota is not None:
            return self.position_quota
        return {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    def normalized_position_score_weights(self) -> dict[str, float]:
        """Return per-position score multipliers for expected value terms."""
        defaults = {"GK": 1.0, "DEF": 1.0, "MID": 1.0, "FWD": 1.0}
        if self.position_score_weights is None:
            return defaults
        merged = defaults.copy()
        merged.update({k: float(v) for k, v in self.position_score_weights.items()})
        return merged

    def normalized_position_risk_weights(self) -> dict[str, float]:
        """Return per-position multipliers for uncertainty penalty terms."""
        defaults = {"GK": 1.0, "DEF": 1.0, "MID": 1.0, "FWD": 1.0}
        if self.position_risk_weights is None:
            return defaults
        merged = defaults.copy()
        merged.update({k: float(v) for k, v in self.position_risk_weights.items()})
        return merged


@dataclass(slots=True)
class TransferContext:
    """Transfer and chip context for a decision window."""

    current_squad_ids: set[str]
    free_transfers: int = 1
    wildcard_available: bool = False
    free_hit_available: bool = False
    bench_boost_available: bool = False
    wildcard_bonus: float = 0.0
    free_hit_bonus: float = 0.0
    bench_boost_bonus: float = 0.0


@dataclass(slots=True)
class SquadPlanResult:
    """Optimizer output for one decision cycle."""

    selected_player_ids: list[str]
    chip_used: str
    expected_points_value: float
    risk_value: float
    transfers_made: float
    paid_transfers: float
    next_free_transfers: int
    objective_value: float


def compute_discounted_value(expected_points: list[float], discount_factor: float) -> float:
    """Compute discounted return across horizon points."""
    return sum((discount_factor**idx) * points for idx, points in enumerate(expected_points))


def compute_discounted_risk(uncertainty: list[float], discount_factor: float) -> float:
    """Compute discounted uncertainty across horizon weeks."""
    return sum((discount_factor**idx) * value for idx, value in enumerate(uncertainty))


def optimize_squad(
    candidates: Iterable[OptimiserInputContract],
    *,
    settings: OptimisationSettings,
    transfer_context: TransferContext,
) -> SquadPlanResult:
    """Solve multi-gameweek FPL squad optimization as a MILP."""
    try:
        import pulp
    except Exception as exc:  # pragma: no cover
        raise ImportError("PuLP is required for optimization. Install pulp in the active env.") from exc

    players = list(candidates)
    if not players:
        raise ValueError("No optimizer candidates provided.")

    quota = settings.normalized_position_quota()
    if sum(quota.values()) != settings.squad_size:
        raise ValueError("Position quota must sum to squad_size.")

    player_idx = list(range(len(players)))

    prob = pulp.LpProblem("fpl_multiweek_planner", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("select", player_idx, lowBound=0, upBound=1, cat="Binary")
    transfer_in = pulp.LpVariable.dicts("transfer_in", player_idx, lowBound=0, upBound=1, cat="Binary")

    chip_names = ["none"]
    if transfer_context.wildcard_available:
        chip_names.append("wildcard")
    if transfer_context.free_hit_available:
        chip_names.append("free_hit")
    if transfer_context.bench_boost_available:
        chip_names.append("bench_boost")
    chip = pulp.LpVariable.dicts("chip", chip_names, lowBound=0, upBound=1, cat="Binary")

    paid_transfers = pulp.LpVariable("paid_transfers", lowBound=0, cat="Continuous")

    # Base squad constraints.
    prob += pulp.lpSum(x[i] for i in player_idx) == settings.squad_size
    prob += pulp.lpSum(chip[name] for name in chip_names) == 1
    prob += pulp.lpSum(players[i].cost * x[i] for i in player_idx) <= settings.budget

    for pos, required in quota.items():
        prob += pulp.lpSum(x[i] for i in player_idx if players[i].position == pos) == required

    teams = sorted({player.team for player in players})
    for team in teams:
        prob += pulp.lpSum(x[i] for i in player_idx if players[i].team == team) <= settings.max_from_team

    # Transfer accounting from current squad.
    current_squad_ids = transfer_context.current_squad_ids
    for i in player_idx:
        in_current = players[i].player_id in current_squad_ids
        if in_current:
            prob += transfer_in[i] == 0
        else:
            prob += transfer_in[i] >= x[i]
            prob += transfer_in[i] <= x[i]

    wildcard_var = chip.get("wildcard", 0)
    free_hit_var = chip.get("free_hit", 0)
    bench_boost_var = chip.get("bench_boost", 0)
    chip_bonus = (
        transfer_context.wildcard_bonus * wildcard_var
        + transfer_context.free_hit_bonus * free_hit_var
        + transfer_context.bench_boost_bonus * bench_boost_var
    )

    transfer_waiver_chip = wildcard_var + free_hit_var
    transfers_made_expr = pulp.lpSum(transfer_in[i] for i in player_idx)
    # Wildcard and Free Hit waive transfer penalties for this optimization step.
    prob += paid_transfers >= transfers_made_expr - transfer_context.free_transfers - settings.squad_size * transfer_waiver_chip
    prob += paid_transfers <= settings.squad_size * (1 - transfer_waiver_chip)

    position_score_weights = settings.normalized_position_score_weights()
    position_risk_weights = settings.normalized_position_risk_weights()

    expected_value = pulp.lpSum(
        x[i]
        * compute_discounted_value(players[i].expected_points_horizon[: settings.horizon_weeks], settings.discount_factor)
        * players[i].availability_probability
        * position_score_weights.get(players[i].position, 1.0)
        for i in player_idx
    )
    risk_value = pulp.lpSum(
        x[i]
        * compute_discounted_risk(players[i].uncertainty_horizon[: settings.horizon_weeks], settings.discount_factor)
        * position_risk_weights.get(players[i].position, 1.0)
        for i in player_idx
    )

    adjustment_value = pulp.lpSum(
        x[i]
        * (
            settings.value_signal_weight * _context_float(players[i], "value_signal")
            - settings.fixture_difficulty_weight * _context_float(players[i], "fixture_difficulty")
            - settings.injury_risk_weight * _context_float(players[i], "injury_risk")
            - settings.rotation_risk_weight * _context_float(players[i], "rotation_risk")
        )
        for i in player_idx
    )

    prob += (
        expected_value
        - settings.risk_aversion * risk_value
        - settings.transfer_penalty * paid_transfers
        + chip_bonus
        + adjustment_value
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Optimization did not find optimal solution: {pulp.LpStatus[prob.status]}")

    selected_indices = [i for i in player_idx if float(x[i].value() or 0.0) > 0.5]
    selected_ids = [players[i].player_id for i in selected_indices]
    selected_chip = next((name for name in chip_names if float(chip[name].value() or 0.0) > 0.5), "none")
    transfers_made = float(sum(1.0 for i in selected_indices if players[i].player_id not in current_squad_ids))
    paid_transfers_value = float(paid_transfers.value() or 0.0)
    next_free_transfers = compute_next_free_transfers(
        current_free_transfers=transfer_context.free_transfers,
        transfers_made=transfers_made,
        chip_used=selected_chip,
    )

    return SquadPlanResult(
        selected_player_ids=selected_ids,
        chip_used=selected_chip,
        expected_points_value=float(pulp.value(expected_value)),
        risk_value=float(pulp.value(risk_value)),
        transfers_made=transfers_made,
        paid_transfers=paid_transfers_value,
        next_free_transfers=next_free_transfers,
        objective_value=float(pulp.value(prob.objective)),
    )


def compute_next_free_transfers(*, current_free_transfers: int, transfers_made: float, chip_used: str) -> int:
    """Compute next gameweek free transfers with carry and cap logic."""
    # Wildcard / Free Hit do not consume transfer bank for next-week carry in this simplified model.
    if chip_used in {"wildcard", "free_hit"}:
        return min(max(current_free_transfers, 1), 2)

    made = int(round(transfers_made))
    if made == 0:
        return min(current_free_transfers + 1, 2)
    return 1


def _context_float(candidate: OptimiserInputContract, key: str) -> float:
    """Read optional numeric adjustment factor from candidate transfer context."""
    value = candidate.transfer_context.get(key, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
