from __future__ import annotations

from pathlib import Path

import pytest

from fpl_ai_agent.config import load_validated_config


def test_load_validated_config_success() -> None:
    cfg = load_validated_config(Path("configs/base.yaml"))
    assert cfg.project.name == "fpl_ai_agent"
    assert cfg.forecasting.window_length >= 1


def test_load_validated_config_invalid_discount(tmp_path: Path) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(
        """
project:
  name: fpl_ai_agent
  timezone: UTC
  random_seed: 42
storage:
  raw_dir: data/raw
  canonical_dir: data/canonical
  features_dir: data/features
  forecasts_dir: data/forecasts
  duckdb_path: data/fpl.duckdb
forecasting:
  window_length: 3
  horizon: 1
  target: target_points
  epochs: 1
  batch_size: 1
  learning_rate: 0.001
  model_version: temporal_cnn_v1
backtesting:
  discount_factor: 1.2
  start_season: 2022
  end_season: 2025
features:
  grain: player-season-gameweek
  window_short: 3
  window_long: 5
  min_history: 3
  availability_column: available_pre_deadline
optimisation:
  horizon_weeks: 3
  budget: 100.0
  transfer_penalty: 4.0
  risk_aversion: 0.15
  squad_size: 15
  max_from_team: 3
  wildcard_bonus: 0.5
  free_hit_bonus: 0.5
  bench_boost_bonus: 0.5
agent:
  discount_factor: 0.98
  transfer_penalty: 4.0
  max_free_transfers: 2
  vice_activation_weight: 0.15
  offline_bootstrap_samples: 200
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid configuration"):
        _ = load_validated_config(bad_cfg)
