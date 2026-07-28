"""Run multi-gameweek squad optimization under FPL constraints."""

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
from fpl_ai_agent.contracts import OptimiserInputContract
from fpl_ai_agent.optimisation.problem import OptimisationSettings, TransferContext, optimize_squad
from fpl_ai_agent.utils.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Optimise squad over multiple gameweeks.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--candidates", default="tests/fixtures/optimiser_candidates.json")
    parser.add_argument("--current-squad", nargs="*", default=[])
    parser.add_argument("--free-transfers", type=int, default=1)
    args = parser.parse_args()

    try:
        cfg = load_validated_config(args.config)
        rows = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("Failed to initialize optimization inputs")
        raise SystemExit(1)

    candidates = [OptimiserInputContract(**row) for row in rows]

    settings = OptimisationSettings(
        horizon_weeks=cfg.optimisation.horizon_weeks,
        discount_factor=cfg.backtesting.discount_factor,
        transfer_penalty=cfg.optimisation.transfer_penalty,
        risk_aversion=cfg.optimisation.risk_aversion,
        budget=cfg.optimisation.budget,
        squad_size=cfg.optimisation.squad_size,
        max_from_team=cfg.optimisation.max_from_team,
    )

    context = TransferContext(
        current_squad_ids=set(args.current_squad),
        free_transfers=args.free_transfers,
        wildcard_available=True,
        free_hit_available=True,
        bench_boost_available=True,
        wildcard_bonus=cfg.optimisation.wildcard_bonus,
        free_hit_bonus=cfg.optimisation.free_hit_bonus,
        bench_boost_bonus=cfg.optimisation.bench_boost_bonus,
    )

    result = optimize_squad(candidates, settings=settings, transfer_context=context)
    print("[optimise_squad] selected_players:", len(result.selected_player_ids))
    print("[optimise_squad] chip_used:", result.chip_used)
    print("[optimise_squad] expected_points_value:", round(result.expected_points_value, 4))
    print("[optimise_squad] risk_value:", round(result.risk_value, 4))
    print("[optimise_squad] paid_transfers:", round(result.paid_transfers, 4))
    print("[optimise_squad] objective_value:", round(result.objective_value, 4))
    print("[optimise_squad] squad_ids:", ",".join(sorted(result.selected_player_ids)))


if __name__ == "__main__":
    main()
