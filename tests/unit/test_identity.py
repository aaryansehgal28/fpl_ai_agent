from __future__ import annotations

import pandas as pd

from fpl_ai_agent.identity.resolution import (
    RapidFuzzResolutionConfig,
    build_canonical_mapping_table,
    resolve_to_canonical,
)


def test_resolve_to_canonical_with_team_position_restrictions() -> None:
    left = pd.DataFrame(
        [
            {"left_id": "fpl_1", "player_name": "Bukayo Saka", "team": "ARS", "position": "MID"},
            {"left_id": "fpl_2", "player_name": "Erling Haaland", "team": "MCI", "position": "FWD"},
        ]
    )
    right = pd.DataFrame(
        [
            {"right_id": "ud_10", "player_name": "B. Saka", "team": "ARS", "position": "MID"},
            {"right_id": "ud_20", "player_name": "E. Haaland", "team": "MCI", "position": "FWD"},
            {"right_id": "ud_99", "player_name": "Bukayo Saka", "team": "MCI", "position": "MID"},
        ]
    )

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

    assert len(matches) == 2
    assert matches[0].left_id == "fpl_1"
    assert matches[0].right_id == "ud_10"
    assert 0.0 <= matches[0].confidence <= 1.0



def test_build_canonical_mapping_table_columns() -> None:
    left = pd.DataFrame(
        [{"left_id": "fpl_1", "player_name": "Cole Palmer", "team": "CHE", "position": "MID"}]
    )
    right = pd.DataFrame(
        [{"right_id": "sf_1", "player_name": "C. Palmer", "team": "CHE", "position": "MID"}]
    )

    matches = resolve_to_canonical(
        left,
        right,
        left_id_col="left_id",
        right_id_col="right_id",
        name_col="player_name",
        team_col="team",
        position_col="position",
        config=RapidFuzzResolutionConfig(score_cutoff=60.0),
    )
    mapping = build_canonical_mapping_table(matches, left_source="fpl", right_source="sofifa")

    expected = {"left_source", "right_source", "left_id", "right_id", "confidence", "method"}
    assert expected.issubset(set(mapping.columns))
    assert mapping.shape[0] == 1
