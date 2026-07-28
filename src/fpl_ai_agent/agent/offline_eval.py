"""Offline policy evaluation utilities for FPL decision policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(slots=True)
class OfflineEvaluationResult:
    """Summary of offline policy evaluation metrics."""

    cumulative_discounted_points: float
    transfer_efficiency: float
    regret_vs_baseline: float
    ci_lower: float
    ci_upper: float


def evaluate_policy(
    *,
    weekly_rewards: list[float],
    transfers_made: list[int],
    baseline_weekly_rewards: list[float],
    discount_factor: float,
    num_bootstrap: int = 200,
    seed: int = 42,
) -> OfflineEvaluationResult:
    """Evaluate a policy trajectory against baseline with bootstrap confidence interval."""
    if not (len(weekly_rewards) == len(transfers_made) == len(baseline_weekly_rewards)):
        raise ValueError("weekly_rewards, transfers_made, and baseline_weekly_rewards must align.")
    if not weekly_rewards:
        return OfflineEvaluationResult(0.0, 0.0, 0.0, 0.0, 0.0)

    discounted_points = _discounted_sum(weekly_rewards, discount_factor)
    baseline_discounted = _discounted_sum(baseline_weekly_rewards, discount_factor)
    regret = baseline_discounted - discounted_points

    total_transfers = sum(transfers_made)
    transfer_eff = discounted_points / max(total_transfers, 1)

    ci_low, ci_high = _bootstrap_ci(
        weekly_rewards=weekly_rewards,
        discount_factor=discount_factor,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )

    return OfflineEvaluationResult(
        cumulative_discounted_points=discounted_points,
        transfer_efficiency=transfer_eff,
        regret_vs_baseline=regret,
        ci_lower=ci_low,
        ci_upper=ci_high,
    )


def _discounted_sum(values: list[float], discount_factor: float) -> float:
    return float(sum((discount_factor**idx) * value for idx, value in enumerate(values)))


def _bootstrap_ci(
    *,
    weekly_rewards: list[float],
    discount_factor: float,
    num_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(weekly_rewards)
    samples: list[float] = []
    for _ in range(num_bootstrap):
        draw = [weekly_rewards[rng.randrange(n)] for _ in range(n)]
        samples.append(_discounted_sum(draw, discount_factor))
    samples.sort()

    lo_idx = max(0, math.floor(0.025 * (len(samples) - 1)))
    hi_idx = min(len(samples) - 1, math.floor(0.975 * (len(samples) - 1)))
    return float(samples[lo_idx]), float(samples[hi_idx])
