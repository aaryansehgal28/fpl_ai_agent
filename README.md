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
```

## Notebooks

- Phase 1 transparency notebook:
	- `notebooks/phase1_scaffold_data_quality.ipynb`

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
