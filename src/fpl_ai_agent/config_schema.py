"""Pydantic schema for validated application configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str
    timezone: str = "UTC"
    random_seed: int = 42


class StorageConfig(BaseModel):
    raw_dir: str
    canonical_dir: str
    features_dir: str
    forecasts_dir: str
    duckdb_path: str


class ForecastingConfig(BaseModel):
    window_length: int = Field(ge=1)
    horizon: int = Field(ge=1)
    target: str
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    model_version: str


class BacktestingConfig(BaseModel):
    discount_factor: float = Field(gt=0, le=1)
    start_season: int
    end_season: int


class FeaturesConfig(BaseModel):
    grain: str
    window_short: int = Field(ge=1)
    window_long: int = Field(ge=1)
    min_history: int = Field(ge=1)
    availability_column: str


class OptimisationConfig(BaseModel):
    horizon_weeks: int = Field(ge=1)
    budget: float = Field(gt=0)
    transfer_penalty: float = Field(ge=0)
    risk_aversion: float = Field(ge=0)
    squad_size: int = Field(ge=1)
    max_from_team: int = Field(ge=1)
    wildcard_bonus: float = 0.0
    free_hit_bonus: float = 0.0
    bench_boost_bonus: float = 0.0


class AgentConfig(BaseModel):
    discount_factor: float = Field(gt=0, le=1)
    transfer_penalty: float = Field(ge=0)
    max_free_transfers: int = Field(ge=0)
    vice_activation_weight: float = Field(ge=0)
    offline_bootstrap_samples: int = Field(ge=1)


class AppConfig(BaseModel):
    project: ProjectConfig
    storage: StorageConfig
    forecasting: ForecastingConfig
    backtesting: BacktestingConfig
    features: FeaturesConfig
    optimisation: OptimisationConfig
    agent: AgentConfig
