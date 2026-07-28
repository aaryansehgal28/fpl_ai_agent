"""Training and evaluation utilities for temporal CNN forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl_ai_agent.evaluation.metrics import mae, rmse
from fpl_ai_agent.features.splitting import FeatureStandardizer, fit_standardizer
from fpl_ai_agent.forecasting.dataset import WindowedDataset, build_temporal_windows
from fpl_ai_agent.forecasting.model import TemporalCNNRegressor


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for training the forecaster.") from exc
    return torch


@dataclass(slots=True)
class ForecastTrainerConfig:
    """Hyperparameters and split setup for forecasting."""

    feature_cols: list[str]
    target_col: str
    window_length: int
    train_seasons: set[int]
    valid_seasons: set[int]
    test_seasons: set[int]
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    model_version: str = "temporal_cnn_v1"


@dataclass(slots=True)
class TrainedForecastBundle:
    """Trained model and preprocessing artifacts."""

    model: object
    standardizer: FeatureStandardizer
    feature_cols: list[str]
    window_length: int
    model_version: str
    metrics: dict[str, float]


def train_forecaster(df: pd.DataFrame, config: ForecastTrainerConfig) -> TrainedForecastBundle:
    """Train temporal CNN with train-only scaling and chronological evaluation."""
    torch = _require_torch()

    train_df = df[df["season"].isin(config.train_seasons)].copy()
    valid_df = df[df["season"].isin(config.valid_seasons)].copy()
    test_df = df[df["season"].isin(config.test_seasons)].copy()

    standardizer = fit_standardizer(train_df, config.feature_cols)
    train_df = _apply_standardizer(train_df, config.feature_cols, standardizer)
    valid_df = _apply_standardizer(valid_df, config.feature_cols, standardizer)
    test_df = _apply_standardizer(test_df, config.feature_cols, standardizer)

    train_data = _build_windows(train_df, config)
    valid_data = _build_windows(valid_df, config)
    test_data = _build_windows(test_df, config)

    model = TemporalCNNRegressor.build(input_features=len(config.feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    x_train = torch.tensor(train_data.x, dtype=torch.float32)
    y_train = torch.tensor(train_data.y, dtype=torch.float32)

    model.train()
    for _ in range(config.epochs):
        for xb, yb in _iterate_minibatches(x_train, y_train, batch_size=config.batch_size):
            pred_mean, pred_var = model(xb)
            loss = 0.5 * torch.mean(torch.log(pred_var) + (yb - pred_mean) ** 2 / pred_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    metrics = _evaluate_model(model, valid_data, test_data)
    return TrainedForecastBundle(
        model=model,
        standardizer=standardizer,
        feature_cols=config.feature_cols,
        window_length=config.window_length,
        model_version=config.model_version,
        metrics=metrics,
    )


def predict_with_uncertainty(
    bundle: TrainedForecastBundle,
    df: pd.DataFrame,
    *,
    target_col: str,
) -> tuple[WindowedDataset, np.ndarray, np.ndarray]:
    """Run model inference and return means and standard deviations."""
    torch = _require_torch()
    transformed = _apply_standardizer(df.copy(), bundle.feature_cols, bundle.standardizer)
    windowed = build_temporal_windows(
        transformed,
        player_col="player_id",
        time_col="gameweek",
        feature_cols=bundle.feature_cols,
        target_col=target_col,
        window_length=bundle.window_length,
    )

    if windowed.x.size == 0:
        return windowed, np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    x = torch.tensor(windowed.x, dtype=torch.float32)
    bundle.model.eval()
    with torch.no_grad():
        pred_mean, pred_var = bundle.model(x)
    means = pred_mean.detach().cpu().numpy()
    stds = torch.sqrt(pred_var).detach().cpu().numpy()
    return windowed, means, stds


def _apply_standardizer(
    df: pd.DataFrame,
    feature_cols: list[str],
    standardizer: FeatureStandardizer,
) -> pd.DataFrame:
    transformed = df.copy()
    for col in feature_cols:
        transformed[col] = (transformed[col] - standardizer.means[col]) / standardizer.stds[col]
    return transformed


def _iterate_minibatches(x, y, *, batch_size: int):
    n = x.shape[0]
    if n == 0:
        return
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield x[start:end], y[start:end]


def _build_windows(df: pd.DataFrame, config: ForecastTrainerConfig) -> WindowedDataset:
    return build_temporal_windows(
        df,
        player_col="player_id",
        time_col="gameweek",
        feature_cols=config.feature_cols,
        target_col=config.target_col,
        window_length=config.window_length,
    )


def _evaluate_model(model, valid: WindowedDataset, test: WindowedDataset) -> dict[str, float]:
    torch = _require_torch()

    def _run_eval(data: WindowedDataset) -> tuple[float, float]:
        if data.x.size == 0:
            return 0.0, 0.0
        x = torch.tensor(data.x, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred_mean, _ = model(x)
        pred = pred_mean.detach().cpu().numpy().tolist()
        y_true = data.y.tolist()
        return mae(y_true, pred), rmse(y_true, pred)

    valid_mae, valid_rmse = _run_eval(valid)
    test_mae, test_rmse = _run_eval(test)
    return {
        "valid_mae": valid_mae,
        "valid_rmse": valid_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }
