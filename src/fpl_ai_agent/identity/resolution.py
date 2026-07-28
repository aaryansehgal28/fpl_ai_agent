"""Entity resolution strategy contracts and fallback fuzzy matching settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IdentityMatch:
    """Represents a candidate identity match across data sources."""

    left_id: str
    right_id: str
    confidence: float
    method: str


@dataclass(slots=True)
class RapidFuzzResolutionConfig:
    """Fallback RapidFuzz strategy with candidate restrictions."""

    restrict_by_team: bool = True
    restrict_by_position: bool = True
    score_cutoff: float = 88.0
