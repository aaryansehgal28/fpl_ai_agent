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
