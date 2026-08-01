from __future__ import annotations

from datetime import datetime

import pytest

from cfb_prediction.config import CFB_MODEL_MANIFEST_PATH
from cfb_prediction.ledger import load_latest_cfb_prediction_batch
from cfb_prediction.modeling import load_cfb_model_bundle
from nfl_prediction.io import sha256_file


def test_published_cfb_forecast_is_pre_game_and_matches_checked_bundle() -> None:
    models, manifest = load_cfb_model_bundle()
    batch = load_latest_cfb_prediction_batch()

    assert set(models) == {"margin", "total"}
    assert batch is not None
    assert batch["model_hash"] == sha256_file(CFB_MODEL_MANIFEST_PATH)
    assert batch["prediction_season"] == 2026
    assert batch["metadata"]["market_data_used"] is False
    assert batch["metadata"]["provisional"] is True
    assert batch["metadata"]["input_coverage"] == manifest["input_coverage"]
    assert manifest["input_coverage"]["returning_production_teams"] == 0
    assert manifest["input_coverage"]["talent_teams"] == 0
    assert len(batch["predictions"]) == 51
    cutoff = datetime.fromisoformat(batch["data_cutoff"])
    assert all(datetime.fromisoformat(row["start_date"]) > cutoff for row in batch["predictions"])

    forbidden = {"market_home_margin", "market_total", "home_points", "away_points"}
    for prediction in batch["predictions"]:
        assert not forbidden.intersection(prediction)
        assert 0.0 <= prediction["home_win_probability"] <= 1.0
        assert prediction["predicted_home_score"] + prediction[
            "predicted_away_score"
        ] == pytest.approx(prediction["predicted_total"])
        assert prediction["predicted_home_score"] - prediction[
            "predicted_away_score"
        ] == pytest.approx(prediction["predicted_home_margin"])

    assert manifest["models"]["margin"]["metrics"]["latest_holdout_mae"] == pytest.approx(
        12.642213058735614
    )
    assert manifest["models"]["total"]["metrics"]["latest_holdout_mae"] == pytest.approx(
        12.730995013066167
    )
