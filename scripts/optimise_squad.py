"""Phase scaffold CLI for squad optimisation."""

from __future__ import annotations

import argparse

from fpl_ai_agent.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimise squad over multiple gameweeks (scaffold).")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    print("[optimise_squad] Loaded config keys:", sorted(config.keys()))
    print("[optimise_squad] Phase 1 scaffold complete. Optimiser logic starts in Phase 5.")


if __name__ == "__main__":
    main()
