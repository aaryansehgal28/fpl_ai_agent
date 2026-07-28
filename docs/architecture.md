# Architecture

This repository is organized into three production layers plus support modules:

1. Forecasting layer: temporal CNN pipeline at player-season-gameweek grain.
2. Optimisation layer: constrained multi-gameweek squad planning.
3. Agent layer: semi-autonomous MDP policy recommendations.

Support modules include ingestion, identity resolution, feature engineering, backtesting, and evaluation.

Phase 1 status: scaffold complete with typed interfaces, contracts, and execution entrypoints.

Phase 2 status: ingestion and canonical mapping baseline implemented.

- Ingestion:
	- Fail-soft source adapters with schema validation and stale detection.
	- Immutable raw JSON snapshots with provenance metadata sidecars.
	- Source orchestration summary with ok/stale/error accounting.

- Identity:
	- Candidate-restricted fuzzy matching (team and/or position first).
	- Confidence-scored canonical mapping table outputs.
	- Point-in-time join helper for temporally correct enrichment.

Phase 3 status: leakage-safe feature store and split logic implemented.

- Feature engineering:
	- Player-season-gameweek grain validation.
	- Pre-deadline availability filtering.
	- Shift-before-roll windows for form and momentum.
	- Advanced proxies: fixture difficulty/ease, expected involvement, injury risk, and value signal.

- Split and preprocessing:
	- Chronological season splits with ordering guardrails.
	- Standardization fit on train-only then applied to valid/test.
