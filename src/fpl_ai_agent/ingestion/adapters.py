"""Source adapters with fail-soft behavior and provenance metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


@dataclass(slots=True)
class SourceFetchResult:
    """Standardized result returned by each source adapter."""

    source_name: str
    pulled_at: datetime
    is_stale: bool
    row_count: int
    payload_path: str
    status: str


class SourceAdapter(Protocol):
    """Protocol for external source ingestion adapters."""

    source_name: str

    def fetch(self) -> SourceFetchResult:
        """Fetch source data and store immutable payload."""


@dataclass(slots=True)
class StaticStubAdapter:
    """Simple adapter used for scaffolding and tests."""

    source_name: str
    payload_path: str

    def fetch(self) -> SourceFetchResult:
        return SourceFetchResult(
            source_name=self.source_name,
            pulled_at=datetime.now(timezone.utc),
            is_stale=False,
            row_count=0,
            payload_path=self.payload_path,
            status="ok",
        )
