"""Run Phase 2 ingestion refresh with fail-soft behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpl_ai_agent.config import load_validated_config
from fpl_ai_agent.ingestion.pipeline import default_source_adapters, run_ingestion
from fpl_ai_agent.utils.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run ingestion and write immutable raw snapshots.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    try:
        cfg = load_validated_config(args.config)
    except Exception:
        LOGGER.exception("Failed to load ingestion configuration")
        raise SystemExit(1)

    raw_dir = cfg.storage.raw_dir
    adapters = default_source_adapters(raw_dir)
    summary = run_ingestion(adapters)

    print("[run_ingestion] total_sources:", summary.total_sources)
    print("[run_ingestion] ok_sources:", summary.ok_sources)
    print("[run_ingestion] stale_sources:", summary.stale_sources)
    print("[run_ingestion] error_sources:", summary.error_sources)


if __name__ == "__main__":
    main()
