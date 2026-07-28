"""Run walk-forward backtest for forecast and decision layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpl_ai_agent.config import load_validated_config
import pandas as pd

from fpl_ai_agent.backtesting.engine import run_walk_forward_backtest
from fpl_ai_agent.contracts import OptimiserInputContract
from fpl_ai_agent.optimisation.problem import OptimisationSettings
from fpl_ai_agent.utils.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run walk-forward backtests.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--raw", default="tests/fixtures/raw/fpl_players_sample.csv")
    parser.add_argument("--candidates", default="tests/fixtures/optimiser_candidates.json")
    parser.add_argument("--output", default="artifacts/backtest/report.json")
    args = parser.parse_args()

    try:
        cfg = load_validated_config(args.config)
        raw_df = pd.read_csv(args.raw)
        candidate_rows = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("Failed to initialize backtest inputs")
        raise SystemExit(1)

    candidate_pool = [OptimiserInputContract(**row) for row in candidate_rows]

    opt_settings = OptimisationSettings(
        horizon_weeks=cfg.optimisation.horizon_weeks,
        discount_factor=cfg.backtesting.discount_factor,
        transfer_penalty=cfg.optimisation.transfer_penalty,
        risk_aversion=cfg.optimisation.risk_aversion,
        budget=cfg.optimisation.budget,
        squad_size=cfg.optimisation.squad_size,
        max_from_team=cfg.optimisation.max_from_team,
    )

    result = run_walk_forward_backtest(
        raw_df=raw_df,
        candidate_pool=candidate_pool,
        optimisation_settings=opt_settings,
        discount_factor=cfg.backtesting.discount_factor,
        transfer_penalty=cfg.optimisation.transfer_penalty,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print("[run_backtest] forecast_metrics:", result.forecast_metrics)
    print("[run_backtest] decision_discounted_points:", round(result.decision_metrics.cumulative_discounted_points, 4))
    print("[run_backtest] decision_regret:", round(result.decision_metrics.regret_vs_baseline, 4))
    print("[run_backtest] decision_ci:", (round(result.decision_metrics.ci_lower, 4), round(result.decision_metrics.ci_upper, 4)))
    print("[run_backtest] weeks_simulated:", len(result.weeks))
    print("[run_backtest] output:", str(output_path))


if __name__ == "__main__":
    main()
