# Post-Build Hardening and Productionization

## Goals
1. Ensure configuration errors fail fast with explicit validation messages.
2. Improve observability with consistent structured logging.
3. Automate code quality and regression checks through CI.
4. Standardize operational release checks.

## Hardening Controls Added
1. Typed configuration schema using Pydantic models in `src/fpl_ai_agent/config_schema.py`.
2. Validated config loader in `src/fpl_ai_agent/config.py` via `load_validated_config`.
3. Root logging bootstrap with optional JSON logs in `src/fpl_ai_agent/utils/logging_utils.py`.
4. Script-level defensive initialization and non-zero exit behavior on startup failures.
5. GitHub Actions workflow for lint, typing, and tests in `.github/workflows/ci.yml`.

## Logging Settings
Set environment variables before running scripts:
1. `FPL_LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
2. `FPL_LOG_JSON`: set to `1` for JSON log lines.

Example:
```bash
FPL_LOG_LEVEL=INFO FPL_LOG_JSON=1 python scripts/run_backtest.py --config configs/base.yaml
```

## Release Gate
Before promoting a build:
1. Run `pytest -q`.
2. Run `ruff check .`.
3. Run `mypy src`.
4. Run `python scripts/run_backtest.py --config configs/base.yaml --raw tests/fixtures/raw/fpl_players_sample.csv --candidates tests/fixtures/optimiser_candidates.json --output artifacts/backtest/report.json`.
5. Verify output artifact exists and key metrics are populated.

## Known Remaining Gaps
1. No secrets manager integration yet; runtime still relies on local environment variables.
2. No container image build pipeline yet.
3. No production scheduler/orchestrator manifest (Airflow, cron, or cloud workflow) committed yet.
