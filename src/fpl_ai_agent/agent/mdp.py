"""MDP primitives and transition logic for semi-autonomous FPL decisions."""

from __future__ import annotations

from dataclasses import dataclass

from fpl_ai_agent.contracts import AgentRecommendationContract


@dataclass(slots=True)
class MDPState:
    """State used by the weekly decision process."""

    gameweek: int
    bank: float
    free_transfers: int
    squad_player_ids: set[str]
    chips_available: dict[str, bool]


@dataclass(slots=True)
class MDPAction:
    """Transfer/chip/lineup action for one gameweek."""

    action_type: str
    payload: dict[str, str | int | float | list[str]]


@dataclass(slots=True)
class Recommendation:
    """Human-approval recommendation output."""

    action: MDPAction
    confidence: float
    rationale: str


@dataclass(slots=True)
class MDPTransitionAssumptions:
    """Assumptions for state transition and reward roll-forward."""

    discount_factor: float = 0.98
    transfer_penalty: float = 4.0
    max_free_transfers: int = 2


@dataclass(slots=True)
class MDPStepResult:
    """Transition result for one step."""

    next_state: MDPState
    reward: float
    expected_value_delta: float
    risk_delta: float


def apply_action(
    state: MDPState,
    action: MDPAction,
    *,
    assumptions: MDPTransitionAssumptions,
) -> MDPStepResult:
    """Apply one weekly action to produce the next state and reward."""
    transfer_in = set(_payload_list(action.payload.get("transfer_in_ids")))
    transfer_out = set(_payload_list(action.payload.get("transfer_out_ids")))
    chip_used = str(action.payload.get("chip", "none"))
    expected_points = float(action.payload.get("expected_points", 0.0))
    risk_penalty = float(action.payload.get("risk_penalty", 0.0))

    # Apply transfers to squad.
    next_squad = set(state.squad_player_ids)
    next_squad -= transfer_out
    next_squad |= transfer_in

    transfers_made = len(transfer_in)
    paid_transfers = max(0, transfers_made - state.free_transfers)
    transfer_cost = assumptions.transfer_penalty * paid_transfers

    reward = expected_points - transfer_cost - risk_penalty
    expected_value_delta = expected_points - transfer_cost
    risk_delta = -risk_penalty

    next_free_transfers = _next_free_transfers(
        free_transfers=state.free_transfers,
        transfers_made=transfers_made,
        chip_used=chip_used,
        max_free_transfers=assumptions.max_free_transfers,
    )

    chips_available = dict(state.chips_available)
    if chip_used in chips_available:
        chips_available[chip_used] = False

    next_state = MDPState(
        gameweek=state.gameweek + 1,
        bank=state.bank,
        free_transfers=next_free_transfers,
        squad_player_ids=next_squad,
        chips_available=chips_available,
    )

    return MDPStepResult(
        next_state=next_state,
        reward=reward,
        expected_value_delta=expected_value_delta,
        risk_delta=risk_delta,
    )


def build_recommendation(
    *,
    action: MDPAction,
    expected_value_delta: float,
    risk_delta: float,
) -> AgentRecommendationContract:
    """Create a semi-autonomous recommendation contract."""
    confidence = _confidence_from_deltas(expected_value_delta, risk_delta)
    explanation = (
        f"Action {action.action_type} proposes expected value delta {expected_value_delta:.2f} "
        f"with risk delta {risk_delta:.2f}."
    )
    return AgentRecommendationContract(
        proposed_action=action.action_type,
        expected_value_delta=expected_value_delta,
        risk_delta=risk_delta,
        confidence=confidence,
        explanation_text=explanation,
    )


def _next_free_transfers(*, free_transfers: int, transfers_made: int, chip_used: str, max_free_transfers: int) -> int:
    if chip_used in {"wildcard", "free_hit"}:
        return min(max(free_transfers, 1), max_free_transfers)
    if transfers_made == 0:
        return min(free_transfers + 1, max_free_transfers)
    return 1


def _payload_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _confidence_from_deltas(expected_value_delta: float, risk_delta: float) -> float:
    numerator = max(expected_value_delta, 0.0)
    denominator = max(abs(expected_value_delta) + abs(risk_delta), 1e-6)
    return max(0.0, min(1.0, numerator / denominator))
