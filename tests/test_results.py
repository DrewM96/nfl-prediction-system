from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_prediction.ledger import PredictionLedger
from nfl_prediction.results import forecast_rows, select_forecasts, settle_schedule, summarize
from nfl_prediction.results_ui import _weekly_chart


def _prediction(game_id: str, margin: float, *, market: float | None = None) -> dict:
    row = {
        "game_id": game_id,
        "season": 2026,
        "week": 1,
        "gameday": "2026-09-13",
        "gametime": "13:00",
        "away_team": "A",
        "home_team": "H",
        "predicted_home_margin": margin,
        "margin_std": 10.0,
        "home_win_probability": 0.6,
        "total": 44.0,
        "total_std": 12.0,
    }
    if market is not None:
        row["market_consensus"] = {
            "snapshot_at": "2026-09-01T12:00:00+00:00",
            "spread": {"market_home_margin": market},
            "total": {"total": 43.0},
        }
    return row


def test_incremental_settlement_is_append_only_and_idempotent(tmp_path: Path) -> None:
    ledger = PredictionLedger(tmp_path)
    batch = ledger.record_batch(
        [_prediction("one", 3), _prediction("two", 20)],
        model_hash="model",
        data_cutoff="2026-09-01",
        prediction_season=2026,
    )
    one = pd.DataFrame([{"game_id": "one", "home_score": 24, "away_score": 21}])
    both = pd.concat(
        [one, pd.DataFrame([{"game_id": "two", "home_score": 20, "away_score": 21}])],
        ignore_index=True,
    )
    assert settle_schedule(tmp_path, one) == 1
    assert settle_schedule(tmp_path, one) == 0
    assert settle_schedule(tmp_path, both) == 1
    assert len(ledger.result_events(batch.stem)) == 2
    assert set(ledger.latest_results(batch.stem)) == {"one", "two"}
    ledger.settle(
        batch.stem,
        [{"game_id": "one", "status": "final", "actual_home_margin": 7, "actual_total": 49}],
    )
    assert len(ledger.result_events(batch.stem)) == 3
    assert ledger.latest_results(batch.stem)["one"]["actual_home_margin"] == 7


def test_comparison_metrics_use_the_identical_market_subset(tmp_path: Path) -> None:
    ledger = PredictionLedger(tmp_path)
    batch = ledger.record_batch(
        [_prediction("one", 0, market=1), _prediction("two", 30)],
        model_hash="model",
        data_cutoff="2026-09-01",
        prediction_season=2026,
    )
    # Make the synthetic batch pregame while keeping the production writer API.
    payload = __import__("json").loads(batch.read_text())
    payload["created_at"] = "2026-09-01T13:00:00+00:00"
    batch.write_text(__import__("json").dumps(payload), encoding="utf-8")
    ledger.settle(
        batch.stem,
        [
            {"game_id": "one", "status": "final", "actual_home_margin": 0, "actual_total": 44},
            {"game_id": "two", "status": "final", "actual_home_margin": 0, "actual_total": 44},
        ],
    )
    rows = select_forecasts(forecast_rows(tmp_path, as_of=datetime(2026, 9, 20, tzinfo=UTC)))
    result = summarize(rows)
    assert result["mae"] == pytest.approx(15)
    assert result["matched_games"] == 1
    assert result["matched_model_mae"] == pytest.approx(0)
    assert result["market_mae"] == pytest.approx(1)
    assert result["winner_games"] == 0  # ties have an explicit exclusion policy


def test_repeated_forecast_runs_count_each_game_once(tmp_path: Path) -> None:
    ledger = PredictionLedger(tmp_path)
    first = ledger.record_batch(
        [_prediction("game", 3)], model_hash="one", data_cutoff="a", prediction_season=2026
    )
    second = ledger.record_batch(
        [_prediction("game", 7)], model_hash="two", data_cutoff="b", prediction_season=2026
    )
    for index, path in enumerate((first, second)):
        payload = __import__("json").loads(path.read_text())
        payload["created_at"] = f"2026-09-0{index + 1}T12:00:00+00:00"
        path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    rows = forecast_rows(tmp_path, as_of=datetime(2026, 9, 12, tzinfo=UTC))
    assert len(rows) == 2
    assert select_forecasts(rows).published_margin.tolist() == [3]
    assert select_forecasts(rows, policy="horizon").published_margin.tolist() == [7]


def test_scored_results_chart_uses_the_pinned_streamlit_api() -> None:
    rows = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "status": "final",
                "published_margin": 3.0,
                "independent_margin": 2.0,
                "market_margin": 1.0,
                "actual_margin": 4.0,
                "published_probability": 0.6,
                "independent_probability": 0.58,
                "published_margin_p10": -10.0,
                "published_margin_p90": 16.0,
                "independent_margin_p10": -11.0,
                "independent_margin_p90": 15.0,
            }
        ]
    )

    _weekly_chart(rows, target="margin")
