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

Phase 4 status: temporal CNN forecasting pipeline implemented.

- Model:
	- PyTorch temporal CNN for next-gameweek points.
	- Predictive mean and variance heads for uncertainty-aware outputs.

- Training and inference:
	- Training utility with train-only scaling and chronological split usage.
	- Inference utility that emits forecast contracts with uncertainty and availability probability.

Phase 5 status: multi-gameweek optimization layer implemented.

- Optimizer:
	- PuLP MILP formulation with legal 15-player constraints.
	- Position quotas, budget cap, and per-team cap.
	- Discounted expected reward objective over horizon weeks.
	- Risk-adjusted objective with uncertainty penalty.
	- Transfer penalties with free-transfer offsets.
	- Chip bonus support for wildcard, free hit, and bench boost.

Phase 6 status: semi-autonomous MDP agent and offline evaluation implemented.

- MDP agent:
	- Explicit state, action, reward, and transition assumptions.
	- Weekly recommendation contract with confidence and rationale.
	- Human-in-the-loop recommendation mode.

- Weekly decision optimizer:
	- Starting XI legality constraints.
	- Bench order optimization.
	- Captain and vice-captain assignment support.

- Offline policy evaluation:
	- Discounted cumulative points.
	- Transfer efficiency and regret versus baseline.
	- Bootstrap confidence intervals for policy value.
