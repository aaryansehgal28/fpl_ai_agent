# Model Cards

This folder will track model cards for:
- temporal CNN forecaster
- optimisation objective variants
- policy models used by the MDP agent

Phase 1 status:
- template placeholders only
- no trained models yet

## Temporal CNN Forecaster (Phase 4)

- Objective: predict next gameweek FPL points per player.
- Input: tensor shaped `(samples, window_length, features)` at player-season-gameweek grain.
- Output: expected points and predictive uncertainty.
- Contract: forecast outputs include expected_points, uncertainty, availability_probability, model_version, generated_at.

Known limitations in current phase:
- Training data fixture is intentionally tiny, so metrics are not production-representative.
- Uncertainty calibration is basic and will be improved during broader backtesting/evaluation phases.
