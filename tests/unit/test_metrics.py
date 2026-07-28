from __future__ import annotations

from fpl_ai_agent.evaluation.metrics import calibration_error, mae, rmse, spearman_rank_correlation


def test_mae_and_rmse() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 3.0, 2.0]
    assert mae(y_true, y_pred) == 2.0 / 3.0
    assert round(rmse(y_true, y_pred), 6) == round((2.0 / 3.0) ** 0.5, 6)


def test_rank_corr_and_calibration() -> None:
    y_true = [2.0, 4.0, 6.0, 8.0]
    y_pred = [2.1, 3.8, 5.7, 7.9]
    y_unc = [1.0, 1.0, 1.0, 1.0]
    assert spearman_rank_correlation(y_true, y_pred) > 0.9
    assert calibration_error(y_true, y_pred, y_unc) >= 0.0
