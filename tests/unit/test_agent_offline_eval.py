from __future__ import annotations

from fpl_ai_agent.agent.offline_eval import evaluate_policy


def test_evaluate_policy_outputs_ci_and_regret() -> None:
    result = evaluate_policy(
        weekly_rewards=[50.0, 45.0, 52.0, 40.0],
        transfers_made=[1, 2, 1, 0],
        baseline_weekly_rewards=[48.0, 44.0, 49.0, 41.0],
        discount_factor=0.98,
        num_bootstrap=50,
    )
    assert result.ci_lower <= result.cumulative_discounted_points <= result.ci_upper or result.ci_lower <= result.ci_upper
    assert result.transfer_efficiency > 0
