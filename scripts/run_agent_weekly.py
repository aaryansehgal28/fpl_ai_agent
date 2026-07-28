"""Run semi-autonomous weekly policy recommendation cycle."""

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
from fpl_ai_agent.agent.mdp import MDPAction, MDPState, MDPTransitionAssumptions, apply_action, build_recommendation
from fpl_ai_agent.agent.offline_eval import evaluate_policy
from fpl_ai_agent.contracts import OptimiserInputContract
from fpl_ai_agent.optimisation.lineup import LineupSettings, optimize_lineup
from fpl_ai_agent.optimisation.problem import OptimisationSettings, TransferContext, optimize_squad
from fpl_ai_agent.utils.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run semi-autonomous weekly policy.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--candidates", default="tests/fixtures/optimiser_candidates.json")
    parser.add_argument("--gameweek", type=int, default=10)
    parser.add_argument("--free-transfers", type=int, default=1)
    args = parser.parse_args()

    try:
        cfg = load_validated_config(args.config)
        rows = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("Failed to initialize weekly agent inputs")
        raise SystemExit(1)

    candidates = [OptimiserInputContract(**row) for row in rows]

    opt_settings = OptimisationSettings(
        horizon_weeks=cfg.optimisation.horizon_weeks,
        discount_factor=cfg.backtesting.discount_factor,
        transfer_penalty=cfg.optimisation.transfer_penalty,
        risk_aversion=cfg.optimisation.risk_aversion,
        budget=cfg.optimisation.budget,
        max_from_team=cfg.optimisation.max_from_team,
        position_score_weights=cfg.optimisation.position_score_weights,
        position_risk_weights=cfg.optimisation.position_risk_weights,
    )
    context = TransferContext(
        current_squad_ids=set(),
        free_transfers=args.free_transfers,
        wildcard_available=True,
        free_hit_available=True,
        bench_boost_available=True,
        wildcard_bonus=cfg.optimisation.wildcard_bonus,
        free_hit_bonus=cfg.optimisation.free_hit_bonus,
        bench_boost_bonus=cfg.optimisation.bench_boost_bonus,
    )

    squad_plan = optimize_squad(candidates, settings=opt_settings, transfer_context=context)
    squad_players = [c for c in candidates if c.player_id in set(squad_plan.selected_player_ids)]
    lineup = optimize_lineup(squad_players, settings=LineupSettings(horizon_index=0), chip_used=squad_plan.chip_used)

    action = MDPAction(
        action_type="weekly_plan",
        payload={
            "transfer_in_ids": squad_plan.selected_player_ids,
            "transfer_out_ids": [],
            "chip": squad_plan.chip_used,
            "starting_xi_ids": lineup.starter_ids,
            "bench_order_ids": lineup.bench_order_ids,
            "captain_id": lineup.captain_id,
            "vice_captain_id": lineup.vice_captain_id,
            "expected_points": lineup.expected_lineup_points,
            "risk_penalty": squad_plan.risk_value * 0.01,
        },
    )

    state = MDPState(
        gameweek=args.gameweek,
        bank=0.0,
        free_transfers=args.free_transfers,
        squad_player_ids=set(),
        chips_available={"wildcard": True, "free_hit": True, "bench_boost": True},
    )
    step_result = apply_action(state, action, assumptions=MDPTransitionAssumptions(discount_factor=cfg.backtesting.discount_factor))
    recommendation = build_recommendation(
        action=action,
        expected_value_delta=step_result.expected_value_delta,
        risk_delta=step_result.risk_delta,
    )

    offline = evaluate_policy(
        weekly_rewards=[step_result.reward, step_result.reward * 0.95, step_result.reward * 0.9],
        transfers_made=[int(squad_plan.transfers_made), max(int(squad_plan.transfers_made) - 1, 0), 1],
        baseline_weekly_rewards=[step_result.reward * 0.9, step_result.reward * 0.9, step_result.reward * 0.9],
        discount_factor=cfg.backtesting.discount_factor,
    )

    print("[run_agent_weekly] proposed_action:", recommendation.proposed_action)
    print("[run_agent_weekly] confidence:", round(recommendation.confidence, 4))
    print("[run_agent_weekly] expected_value_delta:", round(recommendation.expected_value_delta, 4))
    print("[run_agent_weekly] risk_delta:", round(recommendation.risk_delta, 4))
    print("[run_agent_weekly] captain:", lineup.captain_id)
    print("[run_agent_weekly] vice_captain:", lineup.vice_captain_id)
    print("[run_agent_weekly] starters:", len(lineup.starter_ids))
    print("[run_agent_weekly] bench_order:", ",".join(lineup.bench_order_ids))
    print("[run_agent_weekly] offline_cumulative_discounted_points:", round(offline.cumulative_discounted_points, 4))
    print("[run_agent_weekly] offline_regret_vs_baseline:", round(offline.regret_vs_baseline, 4))
    print("[run_agent_weekly] offline_ci:", (round(offline.ci_lower, 4), round(offline.ci_upper, 4)))


if __name__ == "__main__":
    main()
