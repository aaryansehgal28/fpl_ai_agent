"""Train temporal CNN forecaster and export forecast contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpl_ai_agent.config import load_validated_config
from fpl_ai_agent.features.store import build_player_gameweek_features
from fpl_ai_agent.forecasting.inference import build_forecast_contracts
from fpl_ai_agent.forecasting.trainer import ForecastTrainerConfig, predict_with_uncertainty, train_forecaster
from fpl_ai_agent.utils.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train temporal CNN forecaster.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--input", default="tests/fixtures/raw/fpl_players_sample.csv")
    parser.add_argument("--output-dir", default="artifacts/forecasting")
    args = parser.parse_args()

    try:
        cfg = load_validated_config(args.config)
        raw_df = pd.read_csv(args.input)
    except Exception:
        LOGGER.exception("Failed to initialize training inputs")
        raise SystemExit(1)

    feature_df = build_player_gameweek_features(
        raw_df,
        window_short=cfg.features.window_short,
        window_long=cfg.features.window_long,
        min_history=cfg.features.min_history,
    )

    feature_cols = [
        "minutes_prev",
        "goals_prev",
        "assists_prev",
        "ict_prev",
        "bps_prev",
        "price_prev",
        "fixture_difficulty",
        "home_prev",
        "form_points_3",
        "form_points_5",
        "minutes_rolling_3",
        "ict_rolling_3",
        "bps_rolling_3",
        "expected_involvement_proxy",
        "injury_risk_proxy",
        "value_signal",
    ]

    seasons = sorted(feature_df["season"].unique().tolist())
    if len(seasons) >= 3:
        train_seasons = set(seasons[:-2])
        valid_seasons = {seasons[-2]}
        test_seasons = {seasons[-1]}
    else:
        train_seasons = set(seasons)
        valid_seasons = set()
        test_seasons = set()

    trainer_cfg = ForecastTrainerConfig(
        feature_cols=feature_cols,
        target_col="target_points",
        window_length=cfg.forecasting.window_length,
        train_seasons=train_seasons,
        valid_seasons=valid_seasons,
        test_seasons=test_seasons,
        epochs=cfg.forecasting.epochs,
        batch_size=cfg.forecasting.batch_size,
        learning_rate=cfg.forecasting.learning_rate,
        model_version=cfg.forecasting.model_version,
    )

    bundle = train_forecaster(feature_df, trainer_cfg)
    windowed, pred_mean, pred_std = predict_with_uncertainty(
        bundle,
        feature_df,
        target_col="target_points",
    )
    contracts = build_forecast_contracts(
        windowed,
        pred_mean,
        pred_std,
        model_version=bundle.model_version,
        horizon=cfg.forecasting.horizon,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        torch.save(bundle.model.state_dict(), output_dir / "temporal_cnn_state_dict.pt")
    except Exception:
        pass

    model_info = {
        "model_version": bundle.model_version,
        "feature_cols": bundle.feature_cols,
        "window_length": bundle.window_length,
        "metrics": bundle.metrics,
        "standardizer": {
            "means": bundle.standardizer.means,
            "stds": bundle.standardizer.stds,
        },
    }
    (output_dir / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    contracts_rows = [contract.model_dump(mode="json") for contract in contracts]
    (output_dir / "forecast_contracts.json").write_text(
        json.dumps(contracts_rows, indent=2),
        encoding="utf-8",
    )

    print("[train_forecaster] feature_rows:", len(feature_df))
    print("[train_forecaster] train_metrics:", bundle.metrics)
    print("[train_forecaster] contracts_written:", len(contracts_rows))
    print("[train_forecaster] output_dir:", str(output_dir))


if __name__ == "__main__":
    main()
