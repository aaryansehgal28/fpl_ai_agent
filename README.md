# FPL AI Agent

AI-driven Fantasy Premier League platform with a modular three-layer architecture:

1. Layer 1 Forecasting: temporal CNN forecasts at player-season-gameweek grain.
2. Layer 2 Optimisation: constrained multi-gameweek squad planning.
3. Layer 3 Agent: semi-autonomous MDP policy recommendations with offline evaluation.

## Phase-Based Delivery

This repository is built in strict phases and stays runnable at each milestone.

- Phase 1: Scaffold and baseline contracts/configs (complete).
- Phase 2: Ingestion adapters and canonical mapping.
- Phase 3: Feature store and leakage-safe splitting.
- Phase 4: CNN forecasting training/inference pipelines.
- Phase 5: Multi-horizon optimisation.
- Phase 6: MDP agent and offline policy evaluation.
- Phase 8: Post-build hardening and productionization.

## Current Structure

```text
configs/
docs/
notebooks/
scripts/
src/fpl_ai_agent/
tests/
```

## Quickstart

1. Create a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Run tests:

```bash
pytest
```

## Scripts (Scaffold Entry Points)

```bash
python scripts/train_forecaster.py --config configs/base.yaml
python scripts/run_backtest.py --config configs/base.yaml
python scripts/optimise_squad.py --config configs/base.yaml
python scripts/run_agent_weekly.py --config configs/base.yaml
python scripts/run_ingestion.py --config configs/base.yaml
python scripts/build_features.py --config configs/base.yaml
```

## Notebooks

- Phase 1 transparency notebook:
	- `notebooks/phase1_scaffold_data_quality.ipynb`
- Phase 2 transparency notebook:
	- `notebooks/phase2_ingestion_identity_data_quality.ipynb`
- Phase 3 transparency notebook:
	- `notebooks/phase3_feature_store_data_quality.ipynb`
- Phase 4 transparency notebook:
	- `notebooks/phase4_forecasting_data_quality.ipynb`
- Phase 5 transparency notebook:
	- `notebooks/phase5_optimisation_data_quality.ipynb`
- Phase 6 transparency notebook:
	- `notebooks/phase6_agent_data_quality.ipynb`
- Phase 7 transparency notebook:
	- `notebooks/phase7_final_verification_data_quality.ipynb`
- Phase 8 transparency notebook:
	- `notebooks/phase8_post_build_hardening_productionization.ipynb`

The notebook includes:
- what was done
- why it was done
- baseline data quality checks
- anomaly handling notes

## Progress Log

- 2026-07-28: Phase 1 scaffold initialized.
- 2026-07-28: Core contracts added for forecast, optimiser, and agent outputs.
- 2026-07-28: Baseline tests and integration smoke flow added.
- 2026-07-28: Phase 1 data quality notebook added.
- 2026-07-28: Phase 2 ingestion adapters, fail-soft pipeline, and identity resolution implemented.
- 2026-07-28: Phase 2 data quality notebook and tests added.
- 2026-07-28: Phase 3 leakage-safe feature store and chronological splitting implemented.
- 2026-07-28: Phase 3 data quality notebook and tests added.
- 2026-07-28: Phase 4 temporal CNN training and inference pipeline implemented.
- 2026-07-28: Phase 4 data quality notebook and forecasting tests added.
- 2026-07-28: Phase 5 multi-gameweek optimizer with transfer and chip logic implemented.
- 2026-07-28: Phase 5 data quality notebook and optimisation tests added.
- 2026-07-28: Phase 6 MDP agent with semi-autonomous weekly recommendations implemented.
- 2026-07-28: Phase 6 offline policy evaluation and lineup/captain optimization tests added.
- 2026-07-28: Phase 7 final backtesting engine and full verification completed.
- 2026-07-28: Phase 8 hardening completed (validated config, structured logging, CI pipeline, productionization docs).

## Productionization

- CI workflow: `.github/workflows/ci.yml`
- Hardening docs: `docs/productionization.md`

Quality gate command set:

```bash
ruff check .
mypy src
pytest -q
```
