"""Run Phase 2 ingestion refresh with fail-soft behavior."""

from __future__ import annotations

import argparse

from fpl_ai_agent.config import load_yaml_config
from fpl_ai_agent.ingestion.pipeline import default_source_adapters, run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion and write immutable raw snapshots.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    raw_dir = cfg["storage"]["raw_dir"]
    adapters = default_source_adapters(raw_dir)
    summary = run_ingestion(adapters)

    print("[run_ingestion] total_sources:", summary.total_sources)
    print("[run_ingestion] ok_sources:", summary.ok_sources)
    print("[run_ingestion] stale_sources:", summary.stale_sources)
    print("[run_ingestion] error_sources:", summary.error_sources)


if __name__ == "__main__":
    main()
