"""Core data contracts used across forecasting, optimisation, and agent layers."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class ForecastOutputContract(BaseModel):
    """Forecast output at player-gameweek-horizon grain."""

    model_config = ConfigDict(extra="forbid")

    player_id: str
    season: int
    gameweek: int
    horizon: int = Field(ge=1)
    expected_points: float
    uncertainty: float = Field(ge=0)
    availability_probability: float = Field(ge=0, le=1)
    model_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OptimiserInputContract(BaseModel):
    """Optimiser input contract with horizon predictions and risk fields."""

    model_config = ConfigDict(extra="forbid")

    player_id: str
    player_name: str
    team: str
    position: str
    cost: float = Field(ge=0)
    expected_points_horizon: list[float]
    uncertainty_horizon: list[float]
    availability_probability: float = Field(ge=0, le=1)
    transfer_context: dict[str, int | float | str]


class AgentRecommendationContract(BaseModel):
    """Semi-autonomous agent recommendation contract."""

    model_config = ConfigDict(extra="forbid")

    proposed_action: str
    expected_value_delta: float
    risk_delta: float
    confidence: float = Field(ge=0, le=1)
    explanation_text: str
