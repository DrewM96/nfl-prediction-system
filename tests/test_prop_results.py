import copy
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nfl_prediction.io import atomic_write_json
from nfl_prediction.ledger import PredictionLedger
from nfl_prediction.prop_results import (
    freeze_player_predictions,
    latest_prop_results,
    player_breakdown,
    player_forecast_rows,
    prop_events,
    settle_player_predictions,
)

NOW = datetime(2026, 9, 13, 12, tzinfo=UTC)
KICKOFF = datetime(2026, 9, 13, 17, tzinfo=UTC)
AFTER = datetime(2026, 9, 14, 12, tzinfo=UTC)


def projection(**changes):
    return {
        "game_id": "2026_01_PHI_CHI",
        "player_id": "gsis-1",
        "player_name": "Test Player",
        "team": "CHI",
        "opponent": "PHI",
        "category": "passing_yards",
        "projection": 250.0,
        "commence_time": KICKOFF.isoformat(),
        "season": 2026,
        "week": 1,
        "market_line": 240.0,
        "market_at": NOW.isoformat(),
        "market_captured_at": NOW.isoformat(),
        **changes,
    }


def batch(root, name="run", *, published=NOW, projections=None):
    payload = {
        "run_id": name,
        "created_at": published.isoformat(),
        "prediction_season": 2026,
        "model_hash": "model",
        "data_cutoff": "2026-09-10",
        "predictions": [],
        "player_predictions": projections if projections is not None else [projection()],
    }
    atomic_write_json(root / f"{name}.json", payload)
    return root / f"{name}.json"


def schedule():
    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_PHI_CHI",
                "season": 2026,
                "week": 1,
                "home_team": "CHI",
                "away_team": "PHI",
                "commence_time": KICKOFF.isoformat(),
                "home_score": 27,
                "away_score": 7,
            }
        ]
    )


def stats(**changes):
    return pd.DataFrame(
        [
            {
                "player_id": "gsis-1",
                "game_id": "2026_01_PHI_CHI",
                "season": 2026,
                "week": 1,
                "season_type": "REG",
                "team": "CHI",
                "opponent_team": "PHI",
                "passing_yards": 260.0,
                **changes,
            }
        ]
    )


def test_props_are_written_with_game_batch(tmp_path):
    path = PredictionLedger(tmp_path).record_batch(
        [],
        model_hash="x",
        data_cutoff="x",
        prediction_season=2026,
        player_predictions=[projection()],
    )
    assert '"player_predictions"' in path.read_text()


def test_frozen_props_settle_idempotently_and_correct_without_rewriting(tmp_path):
    path = batch(tmp_path)
    original = path.read_bytes()
    assert settle_player_predictions(tmp_path, stats(), schedule(), now=AFTER) == 1
    assert settle_player_predictions(tmp_path, stats(), schedule(), now=AFTER) == 0
    first = player_forecast_rows(tmp_path, as_of=AFTER).iloc[0]
    assert first.result == "Win" and first.error == 10
    assert settle_player_predictions(tmp_path, stats(passing_yards=230), schedule(), now=AFTER) == 1
    second = player_forecast_rows(tmp_path, as_of=AFTER).iloc[0]
    assert second.result == "Loss" and second.error == 20 and second.revision == 2
    assert len(prop_events(tmp_path, "run")) == 2
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "changes",
    [
        {"player_id": "other"},
        {"team": "PHI"},
        {"opponent_team": "DAL"},
        {"game_id": "other"},
        {"passing_yards": None},
        {"season_type": "POST"},
        {"week": 2},
    ],
)
def test_missing_or_mismatched_stats_never_become_zero_or_a_loss(tmp_path, changes):
    batch(tmp_path)
    assert settle_player_predictions(tmp_path, stats(**changes), schedule(), now=AFTER) == 0
    row = player_forecast_rows(tmp_path, as_of=AFTER).iloc[0]
    assert row.status == "awaiting stats" and row.result == "Pending" and pd.isna(row.actual)


def test_published_zero_is_valid_but_ambiguous_stats_are_not(tmp_path):
    batch(tmp_path)
    duplicate = pd.concat([stats(), stats()], ignore_index=True)
    assert settle_player_predictions(tmp_path, duplicate, schedule(), now=AFTER) == 0
    assert settle_player_predictions(tmp_path, stats(passing_yards=0), schedule(), now=AFTER) == 1
    assert player_forecast_rows(tmp_path, as_of=AFTER).iloc[0].actual == 0


def test_in_progress_scores_cannot_settle_props(tmp_path):
    batch(tmp_path)
    assert (
        settle_player_predictions(tmp_path, stats(), schedule(), now=KICKOFF + timedelta(hours=3))
        == 0
    )
    assert not latest_prop_results(tmp_path, "run")


def test_pre_game_selection_deduplicates_and_excludes_late_forecasts(tmp_path):
    batch(tmp_path, "first")
    batch(
        tmp_path,
        "latest",
        published=KICKOFF - timedelta(hours=1),
        projections=[projection(projection=245)],
    )
    batch(
        tmp_path,
        "too-close",
        published=KICKOFF - timedelta(minutes=30),
        projections=[projection(projection=246)],
    )
    batch(
        tmp_path,
        "after",
        published=KICKOFF + timedelta(minutes=1),
        projections=[projection(projection=270)],
    )
    assert player_forecast_rows(tmp_path, as_of=AFTER).projection.tolist() == [250]
    assert player_forecast_rows(tmp_path, policy="horizon", as_of=AFTER).projection.tolist() == [
        245
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"market_line": None},
        {"market_at": (NOW + timedelta(minutes=1)).isoformat()},
        {"market_at": (NOW - timedelta(hours=2)).isoformat()},
        {"market_captured_at": None},
    ],
)
def test_invalid_or_missing_market_line_preserves_projection_accuracy(tmp_path, changes):
    batch(tmp_path, projections=[projection(**changes)])
    settle_player_predictions(tmp_path, stats(), schedule(), now=AFTER)
    row = player_forecast_rows(tmp_path, as_of=AFTER).iloc[0]
    assert row.result == "No line" and row.error == 10 and pd.isna(row.market_line)


def test_push_and_no_pick_are_separate(tmp_path):
    batch(tmp_path, projections=[projection(), projection(player_id="gsis-2", projection=240)])
    actuals = pd.concat([stats(passing_yards=240), stats(player_id="gsis-2", passing_yards=280)])
    settle_player_predictions(tmp_path, actuals, schedule(), now=AFTER)
    rows = player_forecast_rows(tmp_path, as_of=AFTER)
    assert rows.result.tolist() == ["Push", "No pick"]


def test_freezer_uses_only_fresh_consensus_and_does_not_publish_raw_quotes(monkeypatch):
    base = projection()
    for key in ("market_line", "market_at", "market_captured_at"):
        base.pop(key)
    monkeypatch.setattr("nfl_prediction.prop_results.projection_rows", lambda *_: [base])
    quote = {
        "game_id": base["game_id"],
        "commence_time": base["commence_time"],
        "player_key": "testplayer",
        "player_name": "Test Player",
        "team": "CHI",
        "category": "passing_yards",
        "sportsbook": "draftkings",
        "line": 240,
        "over_price": -110,
        "under_price": -110,
        "source_timestamp": NOW.isoformat(),
        "captured_at": NOW.isoformat(),
    }
    board = {"rows": [quote, {**quote, "sportsbook": "fanduel", "line": 250}]}
    depth = {
        "players": [
            {
                "player_id": "gsis-1",
                "player_name": "Test Player",
                "team": "CHI",
                "position": "QB",
                "starter": True,
                "source_timestamp": NOW.isoformat(),
            }
        ]
    }
    snapshot = {"schedule": [{"game_id": base["game_id"], "week": 1}]}
    original = copy.deepcopy(board)
    frozen = freeze_player_predictions(
        snapshot, {}, {"prediction_season": 2026}, board, depth, now=NOW
    )
    assert frozen[0]["market_line"] == 245 and frozen[0]["market_books"] == 2
    assert "quotes" not in frozen[0] and "over_price" not in frozen[0]
    assert board == original
    stale = freeze_player_predictions(
        snapshot, {}, {"prediction_season": 2026}, board, depth, now=NOW + timedelta(hours=2)
    )
    assert stale[0]["market_line"] is None and stale[0]["projection"] == 250
    assert (
        freeze_player_predictions(
            snapshot, {}, {"prediction_season": 2026}, board, depth, now=KICKOFF
        )
        == []
    )


def test_player_results_ui_with_real_archive_and_stats(tmp_path):
    legacy_path = batch(tmp_path, "legacy", published=NOW - timedelta(days=7), projections=[])
    legacy = legacy_path.read_bytes()
    app = AppTest.from_string(
        f'from nfl_prediction.results_ui import _player_results\n_player_results({str(tmp_path)!r}, season=2026, through=1, policy="first", prefix="test")'
    ).run(timeout=15)
    assert not app.exception
    assert any("Earlier releases did not archive" in m.value for m in app.markdown)

    batch(tmp_path)
    settle_player_predictions(tmp_path, stats(), schedule(), now=AFTER)
    app.run(timeout=15)
    assert not app.exception
    assert not any("Earlier releases did not archive" in m.value for m in app.markdown)
    assert legacy_path.read_bytes() == legacy
    assert app.dataframe[0].value.iloc[0]["Result"] == "Win"
    assert app.dataframe[0].value.iloc[0]["GRIDLINE"] == 250
    assert any("1–0–0" in m.value for m in app.markdown)


def test_freezer_runs_existing_models_and_keeps_categories_separate(tmp_path):
    class Model:
        def distribution(self, frame):
            return [{"mean": 25.0, "std": 5.0, "p10": 18.0, "p90": 32.0}]

    player = {
        "prediction_season": 2026,
        "roster_season": 2026,
        "player_name": "Test Player",
        "team": "CHI",
        "opponent": "PHI",
    }
    snapshot = {
        "qb": {"qb-1": player},
        "rb": {"rb-1": player},
        "wr": {"wr-1": player},
        "schedule": schedule().to_dict("records"),
    }
    categories = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
    frozen = freeze_player_predictions(
        snapshot, dict.fromkeys(categories, Model()), {"prediction_season": 2026}, {}, {}, now=NOW
    )
    assert {p["category"] for p in frozen} == set(categories)
    assert len(frozen) == 4 and all(p["projection"] == 25 for p in frozen)
    assert all(p["market_line"] is None for p in frozen)
    batch(tmp_path, projections=frozen)
    rows = player_forecast_rows(tmp_path, as_of=AFTER)
    assert len(rows) == 4 and rows.result.eq("No line").all()


def test_player_report_card_uses_only_settled_rows(tmp_path):
    batch(tmp_path, projections=[projection(), projection(player_id="absent")])
    settle_player_predictions(tmp_path, stats(), schedule(), now=AFTER)
    players = player_breakdown(player_forecast_rows(tmp_path, as_of=AFTER))
    assert len(players) == 1
    assert players.iloc[0].wins == 1 and players.iloc[0].games == 1 and players.iloc[0].mae == 10
