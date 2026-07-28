"""Phase scaffold CLI for training forecaster."""

from __future__ import annotations

import argparse

from fpl_ai_agent.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal CNN forecaster (scaffold).")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    print("[train_forecaster] Loaded config keys:", sorted(config.keys()))
    print("[train_forecaster] Phase 1 scaffold complete. Training logic starts in Phase 4.")


if __name__ == "__main__":
    main()
