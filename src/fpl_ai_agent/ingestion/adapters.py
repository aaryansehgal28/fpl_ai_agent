"""Source adapters with fail-soft behavior and provenance metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.request import urlopen


class SchemaValidationError(ValueError):
    """Raised when source payload does not satisfy contract checks."""


@dataclass(slots=True)
class SourceFetchResult:
    """Standardized result returned by each source adapter."""

    source_name: str
    pulled_at: datetime
    is_stale: bool
    row_count: int
    payload_path: str
    status: str
    message: str = ""


class SourceAdapter(Protocol):
    """Protocol for external source ingestion adapters."""

    source_name: str

    def fetch(self) -> SourceFetchResult:
        """Fetch source data and store immutable payload."""


@dataclass(slots=True)
class SourceSchema:
    """Schema contract used for basic source validation."""

    required_keys: tuple[str, ...]

    def validate_records(self, records: list[dict[str, Any]], *, source_name: str) -> None:
        """Validate required keys for each record."""
        for idx, record in enumerate(records):
            missing = [key for key in self.required_keys if key not in record]
            if missing:
                raise SchemaValidationError(
                    f"{source_name} record {idx} missing required keys: {missing}"
                )


@dataclass(slots=True)
class RawDataStore:
    """Immutable raw storage with provenance sidecar metadata files."""

    root_dir: Path

    def write_json_snapshot(
        self,
        *,
        source_name: str,
        pulled_at: datetime,
        payload: dict[str, Any],
        upstream_ref: str,
    ) -> str:
        """Write immutable JSON payload and provenance metadata."""
        timestamp = pulled_at.strftime("%Y%m%dT%H%M%SZ")
        source_dir = self.root_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        payload_path = source_dir / f"{timestamp}.json"
        metadata_path = source_dir / f"{timestamp}.meta.json"
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        metadata = {
            "source_name": source_name,
            "pulled_at": pulled_at.isoformat(),
            "refresh_timestamp": timestamp,
            "upstream_ref": upstream_ref,
            "payload_path": str(payload_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return str(payload_path)


def _default_json_fetcher(url: str, timeout_seconds: int) -> dict[str, Any]:
    """Fetch JSON from URL using standard library only."""
    with urlopen(url, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


@dataclass(slots=True)
class HTTPJSONSourceAdapter:
    """HTTP JSON adapter with fail-soft semantics and schema checks."""

    source_name: str
    endpoint: str
    schema: SourceSchema
    data_store: RawDataStore
    max_staleness_hours: int = 48
    timeout_seconds: int = 10
    fetcher: Callable[[str, int], dict[str, Any]] = _default_json_fetcher

    def fetch(self) -> SourceFetchResult:
        pulled_at = datetime.now(timezone.utc)
        try:
            payload = self.fetcher(self.endpoint, self.timeout_seconds)
            records = payload.get("records", [])
            if not isinstance(records, list):
                raise SchemaValidationError("Payload field 'records' must be a list.")
            self.schema.validate_records(records, source_name=self.source_name)
            source_updated_at = _extract_updated_at(payload, fallback=pulled_at)
            is_stale = source_updated_at < (pulled_at - timedelta(hours=self.max_staleness_hours))
            payload_path = self.data_store.write_json_snapshot(
                source_name=self.source_name,
                pulled_at=pulled_at,
                payload=payload,
                upstream_ref=self.endpoint,
            )
            status = "stale" if is_stale else "ok"
            message = "source is stale" if is_stale else ""
            return SourceFetchResult(
                source_name=self.source_name,
                pulled_at=pulled_at,
                is_stale=is_stale,
                row_count=len(records),
                payload_path=payload_path,
                status=status,
                message=message,
            )
        except Exception as exc:
            # Fail-soft behavior: return error result and allow pipeline to continue.
            return SourceFetchResult(
                source_name=self.source_name,
                pulled_at=pulled_at,
                is_stale=True,
                row_count=0,
                payload_path="",
                status="error",
                message=str(exc),
            )


@dataclass(slots=True)
class StaticStubAdapter:
    """Simple adapter used for scaffolding and tests."""

    source_name: str
    payload_path: str
    row_count: int = 0
    status: str = "ok"

    def fetch(self) -> SourceFetchResult:
        return SourceFetchResult(
            source_name=self.source_name,
            pulled_at=datetime.now(timezone.utc),
            is_stale=False,
            row_count=self.row_count,
            payload_path=self.payload_path,
            status=self.status,
        )


def _extract_updated_at(payload: dict[str, Any], *, fallback: datetime) -> datetime:
    """Extract refresh timestamp from payload if present, otherwise fallback."""
    value = payload.get("updated_at")
    if not isinstance(value, str):
        return fallback
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return fallback
