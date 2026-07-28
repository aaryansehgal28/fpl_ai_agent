"""Walk-forward backtesting engine for forecast and decision evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from fpl_ai_agent.agent.offline_eval import OfflineEvaluationResult, evaluate_policy
from fpl_ai_agent.contracts import OptimiserInputContract
from fpl_ai_agent.evaluation.metrics import calibration_error, mae, rmse, spearman_rank_correlation
from fpl_ai_agent.features.store import build_player_gameweek_features
from fpl_ai_agent.optimisation.lineup import LineupSettings, optimize_lineup
from fpl_ai_agent.optimisation.problem import OptimisationSettings, TransferContext, optimize_squad


@dataclass(slots=True)
class WeekDecisionResult:
    """Per-gameweek decision summary."""

    gameweek: int
    reward: float
    paid_transfers: float
    transfers_made: float
    chip_used: str
    captain_id: str


@dataclass(slots=True)
class BacktestRunResult:
    """Aggregate backtest outputs."""

    forecast_metrics: dict[str, float]
    decision_metrics: OfflineEvaluationResult
    weeks: list[WeekDecisionResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "forecast_metrics": self.forecast_metrics,
            "decision_metrics": asdict(self.decision_metrics),
            "weeks": [asdict(week) for week in self.weeks],
        }


def run_walk_forward_backtest(
    *,
    raw_df: pd.DataFrame,
    candidate_pool: list[OptimiserInputContract],
    optimisation_settings: OptimisationSettings,
    discount_factor: float,
    transfer_penalty: float,
) -> BacktestRunResult:
    """Run walk-forward backtest over gameweeks with decision simulation."""
    features = build_player_gameweek_features(raw_df, min_history=3)
    if features.empty:
        raise ValueError("Feature frame is empty; cannot run backtest.")

    gameweeks = sorted(features["gameweek"].unique().tolist())
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    y_unc_all: list[float] = []

    current_squad_ids: set[str] = set()
    free_transfers = 1
    weekly_rewards: list[float] = []
    baseline_rewards: list[float] = []
    weekly_transfers: list[int] = []
    week_results: list[WeekDecisionResult] = []

    for gameweek in gameweeks:
        hist = features[features["gameweek"] < gameweek]
        now = features[features["gameweek"] == gameweek]
        if hist.empty or now.empty:
            continue

        # Walk-forward forecast proxy using only pre-gameweek history.
        per_player_hist_mean = (
            hist.groupby("player_id", sort=False)["target_points"].mean().rename("pred_hist_mean")
        )
        joined = now.merge(per_player_hist_mean, how="left", left_on="player_id", right_index=True)
        joined["pred_hist_mean"] = joined["pred_hist_mean"].fillna(joined["form_points_3"])
        joined["pred_uncertainty"] = (joined["form_points_5"] - joined["form_points_3"]).abs().fillna(1.0)

        y_true_all.extend(joined["target_points"].astype(float).tolist())
        y_pred_all.extend(joined["pred_hist_mean"].astype(float).tolist())
        y_unc_all.extend(joined["pred_uncertainty"].astype(float).tolist())

        context = TransferContext(
            current_squad_ids=current_squad_ids,
            free_transfers=free_transfers,
            wildcard_available=True,
            free_hit_available=True,
            bench_boost_available=True,
            wildcard_bonus=0.5,
            free_hit_bonus=0.5,
            bench_boost_bonus=0.5,
        )
        squad_result = optimize_squad(candidate_pool, settings=optimisation_settings, transfer_context=context)
        selected = [c for c in candidate_pool if c.player_id in set(squad_result.selected_player_ids)]
        lineup_result = optimize_lineup(selected, settings=LineupSettings(horizon_index=0), chip_used=squad_result.chip_used)

        reward = lineup_result.expected_lineup_points - transfer_penalty * squad_result.paid_transfers
        weekly_rewards.append(reward)
        baseline_rewards.append(lineup_result.expected_lineup_points * 0.9)
        weekly_transfers.append(int(round(squad_result.transfers_made)))

        week_results.append(
            WeekDecisionResult(
                gameweek=int(gameweek),
                reward=float(reward),
                paid_transfers=float(squad_result.paid_transfers),
                transfers_made=float(squad_result.transfers_made),
                chip_used=squad_result.chip_used,
                captain_id=lineup_result.captain_id,
            )
        )

        current_squad_ids = set(squad_result.selected_player_ids)
        free_transfers = squad_result.next_free_transfers

    forecast_metrics = {
        "mae": mae(y_true_all, y_pred_all),
        "rmse": rmse(y_true_all, y_pred_all),
        "rank_corr": spearman_rank_correlation(y_true_all, y_pred_all),
        "calibration": calibration_error(y_true_all, y_pred_all, y_unc_all),
    }
    decision_metrics = evaluate_policy(
        weekly_rewards=weekly_rewards,
        transfers_made=weekly_transfers,
        baseline_weekly_rewards=baseline_rewards,
        discount_factor=discount_factor,
    )
    return BacktestRunResult(
        forecast_metrics=forecast_metrics,
        decision_metrics=decision_metrics,
        weeks=week_results,
    )
