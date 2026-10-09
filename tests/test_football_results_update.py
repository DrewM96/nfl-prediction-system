from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from cfb_prediction.data import normalize_games
from football_results_update import load_cfb_results, refresh_cfb_results
from nfl_prediction.ledger import PredictionLedger
from nfl_prediction.weekly_games import group_weekly_games
from nfl_results_update import refresh_results

NOW = datetime(2026, 10, 9, 16, tzinfo=UTC)


def _batch(root: Path):
    prediction = {
        "game_id": "one",
        "season": 2026,
        "week": 6,
        "start_date": "2026-10-08T23:00:00Z",
        "gameday": "2026-10-08",
        "gametime": "19:00",
        "home_team": "H",
        "away_team": "A",
        "predicted_home_margin": 3,
        "total": 43,
        "home_win_probability": 0.6,
    }
    path = PredictionLedger(root).record_batch(
        [prediction],
        model_hash="frozen",
        data_cutoff="2026-10-01",
        prediction_season=2026,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["created_at"] = "2026-10-01T12:00:00Z"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, prediction


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_refresh_is_append_only_updates_ui_and_does_not_republish_unchanged_results(
    tmp_path, sport
):
    root = tmp_path / "predictions"
    batch, prediction = _batch(root)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"model": "frozen"}', encoding="utf-8")
    pointer = tmp_path / "latest_prediction.json"
    pointer.write_text(json.dumps({"run_id": batch.stem}), encoding="utf-8")
    frozen = {path: path.read_bytes() for path in (batch, manifest, pointer)}
    schedule = pd.DataFrame(
        [
            {
                "game_id": "one",
                "season": 2026,
                "start_date": "2026-10-08T23:00:00Z",
                "gameday": "2026-10-08",
                "gametime": "19:00",
                "completed": True,
                "home_points": 24,
                "away_points": 17,
                "home_score": 24,
                "away_score": 17,
            }
        ]
    )
    output = tmp_path / "results_summary.json"

    def refresh():
        if sport == "cfb":
            return refresh_cfb_results(schedule, root, output, now=NOW, season=2026)
        return refresh_results(
            schedule, root, output, now=NOW, season=2026, write_if_unchanged=False
        )

    assert refresh()["settlement_documents_added"] == 1
    first_files = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert refresh()["settlement_documents_added"] == 0
    assert first_files == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    results = PredictionLedger(root).latest_results(batch.stem)
    assert results["one"]["actual_home_margin"] == 7
    assert group_weekly_games([prediction], results) == ([], [prediction])
    schedule.loc[0, ["home_points", "home_score"]] = 26
    assert refresh()["settlement_documents_added"] == 1
    assert PredictionLedger(root).latest_results(batch.stem)["one"]["actual_home_margin"] == 9
    assert len(PredictionLedger(root).result_events(batch.stem)) == 2
    for path, original in frozen.items():
        assert path.read_bytes() == original


def test_cfb_in_progress_missing_scores_wrong_season_and_cancelled(tmp_path):
    root = tmp_path / "predictions"
    batch, _ = _batch(root)
    schedule = pd.DataFrame(
        [
            {
                "game_id": "one",
                "season": 2026,
                "completed": False,
                "home_points": 7,
                "away_points": 0,
                "status": "",
            }
        ]
    )
    output = tmp_path / "results_summary.json"
    assert (
        refresh_cfb_results(schedule, root, output, now=NOW, season=2026)[
            "settlement_documents_added"
        ]
        == 0
    )
    assert not output.exists()
    schedule.loc[0, "completed"] = True
    schedule.loc[0, "home_points"] = float("nan")
    assert (
        refresh_cfb_results(schedule, root, output, now=NOW, season=2026)[
            "settlement_documents_added"
        ]
        == 0
    )
    schedule.loc[0, "home_points"] = 24
    assert (
        refresh_cfb_results(schedule, root, output, now=NOW, season=2025)[
            "settlement_documents_added"
        ]
        == 0
    )
    schedule.loc[0, "status"] = "cancelled"
    assert (
        refresh_cfb_results(schedule, root, output, now=NOW, season=2026)[
            "settlement_documents_added"
        ]
        == 1
    )
    result = PredictionLedger(root).latest_results(batch.stem)["one"]
    assert result["status"] == "cancelled" and result["actual_total"] is None


def test_cfb_fetch_is_one_fresh_scores_request():
    class Client:
        def get(self, endpoint, *, params, refresh):
            assert endpoint == "/games"
            assert params == {"year": 2026, "seasonType": "regular", "classification": "fbs"}
            assert refresh is True
            return [{"id": 1, "season": 2026, "completed": True, "homePoints": 0, "awayPoints": 0}]

    scores = load_cfb_results(Client(), 2026)
    assert scores.game_id.tolist() == [1]
    assert bool(scores.completed.iloc[0]) is True
    assert scores.home_points.iloc[0] == 0
    assert normalize_games([]).empty
