"""Core metrics for forecasting and decision evaluation."""

from __future__ import annotations

import math

import pandas as pd


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean absolute error."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred.")
    if not y_true:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root mean squared error."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred.")
    if not y_true:
        return 0.0
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def spearman_rank_correlation(y_true: list[float], y_pred: list[float]) -> float:
    """Spearman rank correlation using rank-transformed Pearson correlation."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred.")
    if len(y_true) < 2:
        return 0.0
    true_rank = pd.Series(y_true).rank(method="average")
    pred_rank = pd.Series(y_pred).rank(method="average")
    corr = true_rank.corr(pred_rank)
    return float(corr) if corr is not None else 0.0


def calibration_error(
    y_true: list[float],
    y_pred: list[float],
    y_uncertainty: list[float],
) -> float:
    """Simple calibration proxy: mean normalized absolute residual."""
    if not (len(y_true) == len(y_pred) == len(y_uncertainty)):
        raise ValueError("Input lengths must match for calibration error.")
    if not y_true:
        return 0.0
    eps = 1e-6
    norm_abs = [abs(t - p) / max(u, eps) for t, p, u in zip(y_true, y_pred, y_uncertainty)]
    return float(sum(norm_abs) / len(norm_abs))
