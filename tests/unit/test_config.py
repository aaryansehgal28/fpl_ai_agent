from __future__ import annotations

from pathlib import Path

from fpl_ai_agent.config import load_yaml_config


def test_load_yaml_config_base() -> None:
    path = Path("configs/base.yaml")
    cfg = load_yaml_config(path)
    assert cfg["project"]["name"] == "fpl_ai_agent"
    assert cfg["forecasting"]["window_length"] == 6
