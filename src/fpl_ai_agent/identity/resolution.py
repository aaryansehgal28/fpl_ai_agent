"""Entity resolution strategy contracts and fallback fuzzy matching settings."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import pandas as pd

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback is covered in tests
    fuzz = None


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


def resolve_to_canonical(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_id_col: str,
    right_id_col: str,
    name_col: str,
    team_col: str,
    position_col: str,
    config: RapidFuzzResolutionConfig | None = None,
) -> list[IdentityMatch]:
    """Resolve left records to right records with candidate restrictions and confidence."""
    cfg = config or RapidFuzzResolutionConfig()
    matches: list[IdentityMatch] = []

    for _, left_row in left_df.iterrows():
        candidates = _restrict_candidates(
            left_row=left_row,
            right_df=right_df,
            team_col=team_col,
            position_col=position_col,
            restrict_by_team=cfg.restrict_by_team,
            restrict_by_position=cfg.restrict_by_position,
        )
        if candidates.empty:
            continue

        best = _best_name_match(
            left_name=str(left_row[name_col]),
            candidates=candidates,
            candidate_name_col=name_col,
        )
        if best is None:
            continue
        best_row, score, method = best
        if score < cfg.score_cutoff:
            continue

        matches.append(
            IdentityMatch(
                left_id=str(left_row[left_id_col]),
                right_id=str(best_row[right_id_col]),
                confidence=score / 100.0,
                method=method,
            )
        )

    return matches


def build_canonical_mapping_table(
    matches: list[IdentityMatch],
    *,
    left_source: str,
    right_source: str,
) -> pd.DataFrame:
    """Build normalized mapping table with confidence and provenance."""
    rows = [
        {
            "left_source": left_source,
            "right_source": right_source,
            "left_id": match.left_id,
            "right_id": match.right_id,
            "confidence": match.confidence,
            "method": match.method,
        }
        for match in matches
    ]
    return pd.DataFrame(rows)


def _restrict_candidates(
    *,
    left_row: pd.Series,
    right_df: pd.DataFrame,
    team_col: str,
    position_col: str,
    restrict_by_team: bool,
    restrict_by_position: bool,
) -> pd.DataFrame:
    """Restrict candidates by team and/or position before fuzzy scoring."""
    filtered = right_df
    if restrict_by_team:
        filtered = filtered[filtered[team_col] == left_row[team_col]]
    if restrict_by_position:
        filtered = filtered[filtered[position_col] == left_row[position_col]]
    return filtered


def _best_name_match(
    *,
    left_name: str,
    candidates: pd.DataFrame,
    candidate_name_col: str,
) -> tuple[pd.Series, float, str] | None:
    """Get highest-scoring candidate using RapidFuzz WRatio or deterministic fallback."""
    best_row: pd.Series | None = None
    best_score = -1.0
    method = ""

    for _, row in candidates.iterrows():
        candidate_name = str(row[candidate_name_col])
        if fuzz is not None:
            score = float(fuzz.WRatio(left_name, candidate_name))
            current_method = "rapidfuzz_wratio"
        else:
            score = _fallback_name_score(left_name, candidate_name)
            current_method = "difflib_ratio"

        if score > best_score:
            best_row = row
            best_score = score
            method = current_method

    if best_row is None:
        return None
    return best_row, best_score, method


def _fallback_name_score(left_name: str, right_name: str) -> float:
    """Score names deterministically with abbreviation-aware heuristics."""
    left_tokens = _name_tokens(left_name)
    right_tokens = _name_tokens(right_name)
    if not left_tokens or not right_tokens:
        return 0.0

    if left_tokens == right_tokens:
        return 100.0

    left_first, left_last = left_tokens[0], left_tokens[-1]
    right_first, right_last = right_tokens[0], right_tokens[-1]

    if left_last == right_last and left_first[:1] == right_first[:1]:
        return 92.0

    left_text = " ".join(left_tokens)
    right_text = " ".join(right_tokens)
    return float(SequenceMatcher(None, left_text, right_text).ratio() * 100)


def _name_tokens(name: str) -> list[str]:
    """Lowercase tokenization that strips punctuation and keeps alphanumerics."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", name.lower()).strip()
    return [token for token in cleaned.split() if token]
