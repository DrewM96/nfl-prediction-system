from pathlib import Path

from cfb_prediction.ledger import (
    load_latest_cfb_prediction_batch,
    record_cfb_prediction_batch,
)


def test_cfb_latest_pointer_resolves_immutable_batch(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    latest = tmp_path / "latest.json"
    target = record_cfb_prediction_batch(
        [{"game_id": 1, "predicted_home_margin": 3.0}],
        model_hash="model-hash",
        data_cutoff="2026-08-01T00:00:00+00:00",
        prediction_season=2026,
        metadata={"forecast_week": 1},
        root=predictions,
        latest_path=latest,
    )

    batch = load_latest_cfb_prediction_batch(root=predictions, latest_path=latest)
    assert batch is not None
    assert batch["run_id"] == target.stem
    assert batch["predictions"][0]["game_id"] == 1
