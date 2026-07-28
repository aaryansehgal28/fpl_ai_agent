"""Data ingestion layer and source adapters."""

from fpl_ai_agent.ingestion.adapters import (
	HTTPJSONSourceAdapter,
	RawDataStore,
	SourceFetchResult,
	SourceSchema,
	StaticStubAdapter,
)
from fpl_ai_agent.ingestion.pipeline import IngestionSummary, default_source_adapters, run_ingestion

__all__ = [
	"HTTPJSONSourceAdapter",
	"RawDataStore",
	"SourceFetchResult",
	"SourceSchema",
	"StaticStubAdapter",
	"IngestionSummary",
	"default_source_adapters",
	"run_ingestion",
]
