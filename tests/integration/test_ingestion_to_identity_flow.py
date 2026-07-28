from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from fpl_ai_agent.identity.resolution import RapidFuzzResolutionConfig, resolve_to_canonical
from fpl_ai_agent.ingestion.adapters import HTTPJSONSourceAdapter, RawDataStore, SourceSchema
from fpl_ai_agent.ingestion.pipeline import run_ingestion


def test_ingestion_to_identity_smoke_flow(tmp_path) -> None:
    schema = SourceSchema(required_keys=("player_name", "team", "position", "source_player_id"))
    store = RawDataStore(root_dir=tmp_path)

    def fpl_fetcher(_: str, __: int) -> dict:
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "records": [
                {
                    "source_player_id": "fpl_1",
                    "player_name": "Bukayo Saka",
                    "team": "ARS",
                    "position": "MID",
                }
            ],
        }

    def understat_fetcher(_: str, __: int) -> dict:
        return {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "records": [
                {
                    "source_player_id": "ud_9",
                    "player_name": "B. Saka",
                    "team": "ARS",
                    "position": "MID",
                }
            ],
        }

    adapters = [
        HTTPJSONSourceAdapter(
            source_name="fpl_official_api",
            endpoint="https://example.com/fpl.json",
            schema=schema,
            data_store=store,
            fetcher=fpl_fetcher,
        ),
        HTTPJSONSourceAdapter(
            source_name="understat",
            endpoint="https://example.com/understat.json",
            schema=schema,
            data_store=store,
            fetcher=understat_fetcher,
        ),
    ]

    summary = run_ingestion(adapters)
    assert summary.ok_sources == 2

    left = pd.DataFrame(fpl_fetcher("", 0)["records"]).rename(columns={"source_player_id": "left_id"})
    right = pd.DataFrame(understat_fetcher("", 0)["records"]).rename(columns={"source_player_id": "right_id"})

    matches = resolve_to_canonical(
        left,
        right,
        left_id_col="left_id",
        right_id_col="right_id",
        name_col="player_name",
        team_col="team",
        position_col="position",
        config=RapidFuzzResolutionConfig(score_cutoff=70.0),
    )
    assert len(matches) == 1
    assert matches[0].left_id == "fpl_1"
