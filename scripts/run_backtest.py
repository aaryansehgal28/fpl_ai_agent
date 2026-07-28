"""Phase scaffold CLI for running backtests."""

from __future__ import annotations

import argparse

from fpl_ai_agent.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward backtests (scaffold).")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    print("[run_backtest] Loaded config keys:", sorted(config.keys()))
    print("[run_backtest] Phase 1 scaffold complete. Backtest engine starts in later phases.")


if __name__ == "__main__":
    main()
