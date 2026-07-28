# Backtesting Protocol

Planned protocol:

1. Walk-forward splits by season and gameweek.
2. Strict anti-leakage feature availability checks.
3. Forecast metrics: MAE, RMSE, rank correlation, calibration.
4. Decision metrics: discounted points, transfer efficiency, and regret.
5. Confidence intervals for all headline outcomes.

Agent policy evaluation additions:
1. Use offline trajectory evaluation before any live automation.
2. Compare policy reward against baseline strategies and report regret.
3. Report bootstrap confidence intervals around discounted policy value.

Phase 1 status: protocol defined, implementation to follow in later phases.
