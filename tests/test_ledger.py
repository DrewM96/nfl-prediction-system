from pathlib import Path

import pytest

from nfl_prediction.ledger import PredictionLedger


def test_prediction_batches_and_results_are_immutable(tmp_path: Path) -> None:
    ledger = PredictionLedger(tmp_path)
    path = ledger.record_batch(
        [{"game_id": "GAME", "predicted_home_margin": 3.0}],
        model_hash="abc",
        data_cutoff="2026-09-08",
        prediction_season=2026,
    )
    run_id = path.stem
    result_path = ledger.score_batch(run_id, [{"game_id": "GAME", "actual_home_margin": 7.0}])
    assert path.exists()
    assert result_path.exists()
    with pytest.raises(FileExistsError):
        ledger.score_batch(run_id, [])
