from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cfb_prediction.modeling import (
    fit_cfb_model,
    load_cfb_model_bundle,
    save_cfb_model_bundle,
)
from nfl_prediction.io import atomic_write_json


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "feature": float(season - 2023 + week),
                "home_margin": float(2 * week - 1),
            }
            for season in (2024, 2025)
            for week in range(1, 5)
        ]
    )


def test_cfb_bundle_round_trips_and_rejects_tampering(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    atomic_write_json(benchmark, {"schema_version": 1})
    model = fit_cfb_model(
        _games(),
        name="margin",
        feature_names=["feature"],
        target_name="home_margin",
        min_train_rows=2,
        alpha=5.0,
    )
    total = fit_cfb_model(
        _games().assign(total_points=lambda frame: frame["home_margin"] + 45),
        name="total",
        feature_names=["feature"],
        target_name="total_points",
        min_train_rows=2,
        alpha=5.0,
    )
    output = tmp_path / "models"
    save_cfb_model_bundle(
        {"margin": model, "total": total},
        prediction_season=2026,
        training_seasons=[2024, 2025],
        data_cutoff="2026-08-01T00:00:00+00:00",
        selected_configurations={"margin": "test", "total": "test"},
        input_coverage={"scheduled_fbs_teams": 2},
        benchmark_path=benchmark,
        output_dir=output,
    )

    loaded, manifest = load_cfb_model_bundle(output / "manifest.json")
    assert manifest["prediction_season"] == 2026
    assert loaded["margin"].predict(pd.DataFrame([{"feature": 3.0}])).shape == (1,)

    with (output / manifest["models"]["margin"]["file"]["path"]).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_cfb_model_bundle(output / "manifest.json")
