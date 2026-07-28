from __future__ import annotations

import pytest

from fpl_ai_agent.contracts import (
    AgentRecommendationContract,
    ForecastOutputContract,
    OptimiserInputContract,
)


def test_forecast_output_contract_valid() -> None:
    contract = ForecastOutputContract(
        player_id="p1",
        season=2024,
        gameweek=3,
        horizon=1,
        expected_points=5.6,
        uncertainty=1.2,
        availability_probability=0.95,
        model_version="v0",
    )
    assert contract.player_id == "p1"


def test_optimiser_input_contract_valid() -> None:
    contract = OptimiserInputContract(
        player_id="p1",
        player_name="Player One",
        team="ARS",
        position="MID",
        cost=7.8,
        expected_points_horizon=[5.0, 5.2, 5.4],
        uncertainty_horizon=[1.0, 1.1, 1.3],
        availability_probability=0.9,
        transfer_context={"free_transfers": 1, "bank": 1.2},
    )
    assert len(contract.expected_points_horizon) == 3


def test_agent_contract_confidence_bounds() -> None:
    with pytest.raises(Exception):
        AgentRecommendationContract(
            proposed_action="roll_transfer",
            expected_value_delta=1.0,
            risk_delta=-0.2,
            confidence=1.2,
            explanation_text="Invalid confidence",
        )
