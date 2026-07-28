# Operations Runbook

## Local Setup
1. Create Python 3.11+ environment.
2. Install package with development extras.
3. Run tests.

## Planned Weekly Cycle
1. Ingest and validate fresh source snapshots.
2. Resolve canonical entities.
3. Build leakage-safe features.
4. Run forecasting inference.
5. Run optimiser.
6. Generate agent recommendation package for human approval.

Phase 1 status: setup and operating skeleton complete.

Phase 2 additions:
1. Run `python scripts/run_ingestion.py --config configs/base.yaml`.
2. Inspect `data/raw/<source>/` for immutable payload and `.meta.json` provenance sidecar files.
3. Review stale/error counts and continue downstream processing only with trusted source slices.

Phase 3 additions:
1. Run `python scripts/build_features.py --config configs/base.yaml`.
2. Confirm output is one row per player-season-gameweek after filtering.
3. Confirm pre-deadline-only rows and no missing required model feature fields.

Phase 5 additions:
1. Prepare optimizer candidate inputs from forecast contracts and metadata.
2. Run `python scripts/optimise_squad.py --config configs/base.yaml --candidates tests/fixtures/optimiser_candidates.json`.
3. Validate legal squad counts, budget, team caps, and transfer/chip summary outputs.

Phase 6 additions:
1. Run `python scripts/run_agent_weekly.py --config configs/base.yaml --candidates tests/fixtures/optimiser_candidates.json --gameweek 10 --free-transfers 1`.
2. Review recommended action, confidence, rationale, lineup, bench order, and captain/vice choices.
3. Review offline evaluation summary (discounted points, regret vs baseline, confidence interval) before operational use.

Phase 7 additions:
1. Run `python scripts/run_backtest.py --config configs/base.yaml --raw tests/fixtures/raw/fpl_players_sample.csv --candidates tests/fixtures/optimiser_candidates.json --output artifacts/backtest/report.json`.
2. Validate forecast and decision metrics in the generated report.
3. Confirm all test suites pass before release.

Phase 8 additions:
1. Validate configuration schema by running any script with `--config configs/base.yaml`; startup must fail fast on invalid config.
2. Optionally enable structured logs via `FPL_LOG_JSON=1` and set explicit log level with `FPL_LOG_LEVEL`.
3. Run release gate checks: `ruff check .`, `mypy src`, `pytest -q`.
4. Confirm CI workflow `.github/workflows/ci.yml` passes on pull requests before merge.
