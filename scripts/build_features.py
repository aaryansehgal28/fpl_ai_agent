"""Build leakage-safe feature store dataset for modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpl_ai_agent.config import load_yaml_config
from fpl_ai_agent.features.store import build_player_gameweek_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build player-season-gameweek features.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--input", default="tests/fixtures/raw/fpl_players_sample.csv")
    parser.add_argument("--output", default="data/features/player_gameweek_features.csv")
    args = parser.parse_args()

    _ = load_yaml_config(args.config)
    raw_df = pd.read_csv(args.input)
    features = build_player_gameweek_features(raw_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print("[build_features] rows:", len(features))
    print("[build_features] output:", str(output_path))


if __name__ == "__main__":
    main()
