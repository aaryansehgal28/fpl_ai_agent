from __future__ import annotations

from fpl_ai_agent.ingestion.adapters import StaticStubAdapter


def test_static_stub_adapter_fetch_ok() -> None:
    adapter = StaticStubAdapter(source_name="fpl_api", payload_path="data/raw/fpl_api/test.json")
    result = adapter.fetch()
    assert result.source_name == "fpl_api"
    assert result.status == "ok"
