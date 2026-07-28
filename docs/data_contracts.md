# Data Contracts

## Forecast Output Contract
Grain: player-gameweek-horizon

Required fields:
- expected_points
- uncertainty
- availability_probability
- model_version
- generated_at

Implemented in `fpl_ai_agent.contracts.ForecastOutputContract`.

## Optimiser Input Contract
Required fields:
- player metadata
- cost
- position
- team
- horizon predictions
- risk fields
- transfer context

Implemented in `fpl_ai_agent.contracts.OptimiserInputContract`.

## Agent Recommendation Contract
Required fields:
- proposed_action
- expected_value_delta
- risk_delta
- confidence
- explanation_text

Implemented in `fpl_ai_agent.contracts.AgentRecommendationContract`.
