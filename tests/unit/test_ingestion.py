from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fpl_ai_agent.ingestion.adapters import (
    HTTPJSONSourceAdapter,
    RawDataStore,
    SourceSchema,
    StaticStubAdapter,
)
from fpl_ai_agent.ingestion.pipeline import run_ingestion


def test_static_stub_adapter_fetch_ok() -> None:
    adapter = StaticStubAdapter(source_name="fpl_api", payload_path="data/raw/fpl_api/test.json")
    result = adapter.fetch()
    assert result.source_name == "fpl_api"
    assert result.status == "ok"


def test_http_adapter_ok_with_schema_and_storage(tmp_path) -> None:
    store = RawDataStore(root_dir=tmp_path)
    schema = SourceSchema(required_keys=("player_name", "team", "position"))

    def fetcher(_: str, __: int) -> dict:
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "records": [
                {"player_name": "Bukayo Saka", "team": "ARS", "position": "MID"},
            ],
        }

    adapter = HTTPJSONSourceAdapter(
        source_name="fpl_official_api",
        endpoint="https://example.com/source.json",
        schema=schema,
        data_store=store,
        fetcher=fetcher,
    )
    result = adapter.fetch()
    assert result.status == "ok"
    assert result.row_count == 1
    assert result.payload_path


def test_http_adapter_stale_source(tmp_path) -> None:
    store = RawDataStore(root_dir=tmp_path)
    schema = SourceSchema(required_keys=("player_name", "team", "position"))
    stale_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

    def fetcher(_: str, __: int) -> dict:
        return {
            "updated_at": stale_time,
            "records": [{"player_name": "Old Data", "team": "ARS", "position": "MID"}],
        }

    adapter = HTTPJSONSourceAdapter(
        source_name="understat",
        endpoint="https://example.com/source.json",
        schema=schema,
        data_store=store,
        max_staleness_hours=1,
        fetcher=fetcher,
    )
    result = adapter.fetch()
    assert result.status == "stale"
    assert result.is_stale


def test_http_adapter_fail_soft_on_schema_error(tmp_path) -> None:
    store = RawDataStore(root_dir=tmp_path)
    schema = SourceSchema(required_keys=("player_name", "team", "position"))

    def fetcher(_: str, __: int) -> dict:
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "records": [{"player_name": "Missing team and position"}],
        }

    adapter = HTTPJSONSourceAdapter(
        source_name="sofifa",
        endpoint="https://example.com/source.json",
        schema=schema,
        data_store=store,
        fetcher=fetcher,
    )
    result = adapter.fetch()
    assert result.status == "error"
    assert result.is_stale
    assert "missing required keys" in result.message


def test_run_ingestion_summary_counts() -> None:
    adapters = [
        StaticStubAdapter(source_name="a", payload_path="/tmp/a.json", status="ok"),
        StaticStubAdapter(source_name="b", payload_path="/tmp/b.json", status="ok"),
        StaticStubAdapter(source_name="c", payload_path="/tmp/c.json", status="stale"),
        StaticStubAdapter(source_name="d", payload_path="/tmp/d.json", status="error"),
    ]
    summary = run_ingestion(adapters)
    assert summary.total_sources == 4
    assert summary.ok_sources == 2
    assert summary.stale_sources == 1
    assert summary.error_sources == 1
