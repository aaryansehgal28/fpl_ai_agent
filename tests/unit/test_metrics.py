from __future__ import annotations

from fpl_ai_agent.evaluation.metrics import mae, rmse


def test_mae_and_rmse() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 3.0, 2.0]
    assert mae(y_true, y_pred) == 2.0 / 3.0
    assert round(rmse(y_true, y_pred), 6) == round((2.0 / 3.0) ** 0.5, 6)
