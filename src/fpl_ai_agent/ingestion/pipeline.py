"""Ingestion orchestration across all configured sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fpl_ai_agent.ingestion.adapters import (
    HTTPJSONSourceAdapter,
    RawDataStore,
    SourceAdapter,
    SourceFetchResult,
    SourceSchema,
)


@dataclass(slots=True)
class IngestionSummary:
    """Aggregate outcomes of one ingestion refresh run."""

    total_sources: int
    ok_sources: int
    stale_sources: int
    error_sources: int
    results: list[SourceFetchResult]


def default_source_adapters(raw_root_dir: str | Path) -> list[SourceAdapter]:
    """Build source adapters for all required external data providers."""
    store = RawDataStore(root_dir=Path(raw_root_dir))
    shared_schema = SourceSchema(required_keys=("player_name", "team", "position"))

    return [
        HTTPJSONSourceAdapter(
            source_name="fpl_official_api",
            endpoint="https://fantasy.premierleague.com/api/bootstrap-static/",
            schema=shared_schema,
            data_store=store,
        ),
        HTTPJSONSourceAdapter(
            source_name="fpl_historical_github",
            endpoint="https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv",
            schema=shared_schema,
            data_store=store,
        ),
        HTTPJSONSourceAdapter(
            source_name="clubelo",
            endpoint="https://api.clubelo.com/2025-01-01",
            schema=shared_schema,
            data_store=store,
        ),
        HTTPJSONSourceAdapter(
            source_name="sofifa",
            endpoint="https://sofifa.com/players",
            schema=shared_schema,
            data_store=store,
        ),
        HTTPJSONSourceAdapter(
            source_name="understat",
            endpoint="https://understat.com/league/EPL/2025",
            schema=shared_schema,
            data_store=store,
        ),
        HTTPJSONSourceAdapter(
            source_name="injury_source",
            endpoint="https://example.com/whoscored_style_injuries.json",
            schema=shared_schema,
            data_store=store,
        ),
    ]


def run_ingestion(adapters: list[SourceAdapter]) -> IngestionSummary:
    """Run all adapters with fail-soft behavior and return summary stats."""
    results = [adapter.fetch() for adapter in adapters]
    ok_sources = sum(result.status == "ok" for result in results)
    stale_sources = sum(result.status == "stale" for result in results)
    error_sources = sum(result.status == "error" for result in results)

    return IngestionSummary(
        total_sources=len(results),
        ok_sources=ok_sources,
        stale_sources=stale_sources,
        error_sources=error_sources,
        results=results,
    )
