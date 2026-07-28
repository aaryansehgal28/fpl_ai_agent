from __future__ import annotations

import pytest


def test_train_forecaster_script_runs(tmp_path) -> None:
    pytest.importorskip("torch", reason="torch required")

    from scripts.train_forecaster import main
    import sys

    output_dir = tmp_path / "forecasting"
    argv = [
        "train_forecaster.py",
        "--config",
        "configs/base.yaml",
        "--input",
        "tests/fixtures/raw/fpl_players_sample.csv",
        "--output-dir",
        str(output_dir),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old_argv

    assert (output_dir / "model_info.json").exists()
    assert (output_dir / "forecast_contracts.json").exists()
