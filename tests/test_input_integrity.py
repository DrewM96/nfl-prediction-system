from datetime import UTC, datetime

import pandas as pd
import pytest

from nfl_prediction.features import build_point_in_time_game_features
from nfl_prediction.io import read_json
from nfl_prediction.ledger import PredictionLedger
from nfl_prediction.pipeline import run_update
from nfl_prediction.preseason import apply_preseason_calibration
from nfl_prediction.quality import forecast_quality, validate_source_schema
from nfl_results_update import refresh_results


def test_snapshot_and_next_scheduled_game_have_identical_team_features():
    schedules = pd.DataFrame(
        [
            dict(
                game_id="past",
                season=2025,
                week=1,
                gameday="2025-09-07",
                home_team="A",
                away_team="B",
                home_score=40,
                away_score=10,
            ),
            dict(
                game_id="next",
                season=2025,
                week=2,
                gameday="2025-09-14",
                home_team="B",
                away_team="A",
                home_score=None,
                away_score=None,
            ),
        ]
    )
    result = build_point_in_time_game_features(schedules, pd.DataFrame(), include_unplayed=True)
    future = result.games.iloc[-1]
    for side, team in (("home", "B"), ("away", "A")):
        for key, value in result.team_snapshot[team].items():
            assert future[f"{side}_{key}"] == pytest.approx(value)
    assert future.input_audit["home"]["missing_pbp_games_l4"] == ["past"]
    assert forecast_quality(future, True)["status"] == "blocked"


def test_source_schema_cannot_silently_replace_missing_epa_with_zero():
    with pytest.raises(ValueError, match="epa"):
        validate_source_schema(pd.DataFrame(), pd.DataFrame({"game_id": ["one"]}))


def test_present_pbp_with_missing_epa_is_not_treated_as_observed_zero():
    schedule = pd.DataFrame(
        [
            dict(
                game_id="past",
                season=2026,
                week=1,
                gameday="2026-09-01",
                home_team="A",
                away_team="B",
                home_score=20,
                away_score=17,
            ),
            dict(
                game_id="future",
                season=2026,
                week=2,
                gameday="2026-09-08",
                home_team="A",
                away_team="B",
                home_score=None,
                away_score=None,
            ),
        ]
    )
    plays = pd.DataFrame(
        [
            dict(
                game_id="past",
                posteam=team,
                defteam=opponent,
                play_type="pass",
                epa=None,
                qb_dropback=1,
                yards_gained=5,
            )
            for team, opponent in (("A", "B"), ("B", "A"))
        ]
    )
    game = build_point_in_time_game_features(schedule, plays, include_unplayed=True).games.iloc[-1]
    quality = forecast_quality(game, False)
    assert quality["status"] == "blocked"
    assert any("no valid" in reason for reason in quality["failures"])
    assert game.input_audit["home"]["sources_l8"][0]["missing_epa_plays"] == 2


def test_historical_live_update_is_rejected_before_fetching(monkeypatch):
    def fail_if_called(*args):
        raise AssertionError("Must reject the replay before downloading live feeds")

    monkeypatch.setattr("nfl_prediction.pipeline.load_nflverse_data", fail_if_called)
    with pytest.raises(ValueError, match="Historical updates"):
        run_update(datetime(2020, 1, 1, tzinfo=UTC))


def test_sparse_market_explicitly_reports_football_fallback():
    prediction = dict(home_team="A", away_team="B", week=1, predicted_home_margin=2.0)
    market = {
        "snapshot_at": "2026-09-01T00:00:00Z",
        "games": [
            dict(
                home_team="A",
                away_team="B",
                commence_time="2026-09-05T17:00:00Z",
                spread=dict(market_home_margin=5, book_count=3),
            )
        ],
    }
    result = apply_preseason_calibration(
        [prediction], market, as_of=datetime(2026, 9, 2, tzinfo=UTC)
    )[0]
    assert result["predicted_home_margin"] == 2
    assert result["forecast_method"] == "football only"
    assert result["calibration_status"]["fallback_reason"] == "insufficient_market_schedule_rank"


def test_settlement_preserves_source_and_does_not_settle_live_scores(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger")
    batch = ledger.record_batch(
        [
            dict(
                game_id="past",
                season=2026,
                week=1,
                gameday="2026-09-01",
                gametime="13:00",
                home_team="A",
                away_team="B",
                predicted_home_margin=3,
                total=44,
            ),
            dict(
                game_id="live",
                season=2026,
                week=1,
                gameday="2026-09-02",
                gametime="13:00",
                home_team="A",
                away_team="B",
                predicted_home_margin=3,
                total=44,
            ),
        ],
        model_hash="model",
        data_cutoff="2026-08-31",
        prediction_season=2026,
    )
    original = batch.read_bytes()
    schedule = pd.DataFrame(
        [
            dict(
                game_id="past", gameday="2026-09-01", gametime="13:00", home_score=24, away_score=20
            ),
            dict(
                game_id="live", gameday="2026-09-02", gametime="13:00", home_score=7, away_score=0
            ),
        ]
    )
    report = refresh_results(
        schedule, ledger.root, tmp_path / "summary.json", now=datetime(2026, 9, 2, 18, tzinfo=UTC)
    )
    assert report["settlement_documents_added"] == 1
    assert set(ledger.latest_results(batch.stem)) == {"past"}
    assert ledger.result_events(batch.stem)[0]["source"] == "nflverse schedules"
    assert batch.read_bytes() == original
    refresh_results(
        schedule, ledger.root, tmp_path / "summary.json", now=datetime(2026, 9, 2, 18, tzinfo=UTC)
    )
    assert len(ledger.result_events(batch.stem)) == 1
    assert read_json(tmp_path / "summary.json")["settlement_documents_added"] == 0
