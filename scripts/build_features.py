"""Build leakage-safe feature store dataset for modeling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpl_ai_agent.config import load_validated_config
from fpl_ai_agent.features.store import build_player_gameweek_features
from fpl_ai_agent.utils.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Build player-season-gameweek features.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--input", default="tests/fixtures/raw/fpl_players_sample.csv")
    parser.add_argument("--output", default="data/features/player_gameweek_features.csv")
    args = parser.parse_args()

    try:
        _ = load_validated_config(args.config)
        raw_df = pd.read_csv(args.input)
    except Exception:
        LOGGER.exception("Failed to initialize feature build inputs")
        raise SystemExit(1)

    features = build_player_gameweek_features(raw_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print("[build_features] rows:", len(features))
    print("[build_features] output:", str(output_path))


if __name__ == "__main__":
    main()
