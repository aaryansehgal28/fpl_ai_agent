"""Phase scaffold CLI for weekly agent recommendations."""

from __future__ import annotations

import argparse

from fpl_ai_agent.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semi-autonomous weekly policy (scaffold).")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    print("[run_agent_weekly] Loaded config keys:", sorted(config.keys()))
    print("[run_agent_weekly] Phase 1 scaffold complete. Agent logic starts in Phase 6.")


if __name__ == "__main__":
    main()
