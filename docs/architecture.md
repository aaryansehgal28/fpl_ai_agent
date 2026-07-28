# Architecture

This repository is organized into three production layers plus support modules:

1. Forecasting layer: temporal CNN pipeline at player-season-gameweek grain.
2. Optimisation layer: constrained multi-gameweek squad planning.
3. Agent layer: semi-autonomous MDP policy recommendations.

Support modules include ingestion, identity resolution, feature engineering, backtesting, and evaluation.

Phase 1 status: scaffold complete with typed interfaces, contracts, and execution entrypoints.
