from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_cfb_features import _historical

import cfb_ranking_evaluation as evaluation
from cfb_prediction.data import CFBHistoricalData
from cfb_ranking_evaluation import (
    common_opponent_ratings,
    freeze_data,
    frozen_frames,
    paired_intervals,
    results_ratings,
)


class LinearModel:
    def predict(self, frame):
        # Deliberately asymmetric designated-home and away coefficients.
        return (2 * frame.home_elo - frame.away_elo + 7 * frame.week + 20).to_numpy()


def test_common_opponent_symmetry_removes_role_bias_and_preserves_strength():
    snapshots = {"A": {"elo": 1500.0}, "B": {"elo": 1510.0}, "C": {"elo": 1490.0}}
    ratings = common_opponent_ratings(LinearModel(), snapshots, week=3)
    reverse = common_opponent_ratings(
        LinearModel(), dict(reversed(list(snapshots.items()))), week=8
    )
    assert ratings == pytest.approx({"A": 0.0, "B": 15.0, "C": -15.0})
    assert ratings == pytest.approx(reverse)


def test_freeze_excludes_every_same_week_result_even_if_kickoff_was_earlier():
    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "season": [2025] * 3,
            "week": [1, 2, 2],
            "start_date": pd.to_datetime(["2025-09-01", "2025-09-08", "2025-09-09"], utc=True),
            "completed": [True] * 3,
            "home_points": [20, 90, 70],
            "away_points": [10, 1, 2],
        }
    )
    advanced = pd.DataFrame({"game_id": [1, 2, 3], "off_ppa": [0.1, 9.0, 7.0]})
    empty = pd.DataFrame()
    data = CFBHistoricalData(games, advanced, empty, empty, empty, empty, empty)
    frozen = freeze_data(data, 2025, 2, pd.Timestamp("2025-09-09", tz="UTC"))
    assert frozen.games.completed.tolist() == [True, False, False]
    assert frozen.games.home_points.iloc[1:].isna().all()
    assert frozen.advanced.game_id.tolist() == [1]
    assert data.games.completed.all()


def test_results_keep_prior_baseline_between_disconnected_groups():
    games = pd.DataFrame(
        {
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "completed": [True, True],
            "fbs_vs_fbs": [True, True],
            "home_points": [20, 20],
            "away_points": [10, 10],
            "neutral_site": [True, True],
        }
    )
    prior = {"A": 10.0, "B": 10.0, "C": -10.0, "D": -10.0}
    ratings = results_ratings(games, prior, prior, 4.0, 28)
    assert np.mean([ratings["A"], ratings["B"]]) == pytest.approx(10.0)
    assert np.mean([ratings["C"], ratings["D"]]) == pytest.approx(-10.0)


def test_paired_bootstrap_reports_signed_error_difference():
    rows = [
        {"season": 2025, "week": w, "actual": 10, "candidate": 12, "schedule_projection": 15}
        for w in range(1, 5)
    ]
    result = paired_intervals(rows, "candidate", samples=100)
    assert result["delta_mae"] == pytest.approx(-3.0)
    assert result["ci95"] == pytest.approx([-3.0, -3.0])


def test_snapshot_replay_ignores_validation_scores_and_efficiency():
    data = _historical()
    data.games["home_conference"] = "SEC"
    data.games["away_conference"] = "ACC"
    cutoff = pd.Timestamp("2025-09-06T16:00:00Z")
    _, _, before, _ = frozen_frames(data, 2025, 2, cutoff)
    data.games.loc[data.games.week.eq(2), "home_points"] = 99
    data.advanced.loc[data.advanced.game_id.eq(2), "off_ppa"] = 100.0
    _, _, after, _ = frozen_frames(data, 2025, 2, cutoff)
    assert before == after


def test_complete_benchmark_serializes_and_keeps_holdout_out_of_selection(monkeypatch, tmp_path):
    parts = []
    for season in range(2018, 2027):
        data = _historical()
        frames = {}
        for name in data.__dataclass_fields__:
            frame = getattr(data, name).copy()
            if "season" in frame:
                frame["season"] = season
            if "game_id" in frame:
                frame["game_id"] = frame.game_id + season * 10
            for column in ("start_date", "transfer_date"):
                if column in frame:
                    frame[column] = frame[column].map(
                        lambda value, year=season: value.replace(year=year)
                    )
            frame = frame.replace({"Alpha": "James Madison", "Beta": "Other"})
            frames[name] = frame
        frames["games"]["home_conference"] = "Sun Belt"
        frames["games"]["away_conference"] = "American"
        parts.append(replace(data, **frames))
    data = evaluation._combine(parts)
    monkeypatch.setattr(evaluation.CFBDClient, "from_environment", lambda: object())

    def fake_load(client, seasons, **kwargs):
        return replace(
            data,
            **{
                name: frame[frame.season.isin(seasons)].copy()
                if "season" in frame
                else frame.iloc[:0].copy()
                for name in data.__dataclass_fields__
                for frame in [getattr(data, name)]
            },
        )

    monkeypatch.setattr(evaluation, "load_historical_data", fake_load)
    output = tmp_path / "evaluation.json"
    evaluation.run(output)
    result = json.loads(output.read_text())
    assert result["metrics"]["development_2022_2024"]["all"]["schedule_projection"]["games"] == 3
    assert result["metrics"]["holdout_2025"]["all"]["schedule_projection"]["games"] == 1
    assert result["metrics"]["followup_2026"]["all"]["schedule_projection"]["games"] == 1
    assert all(m["games"] == 3 for m in result["candidate_development_metrics"].values())
    assert len(result["skipped_weeks"]) == 5
    for method in evaluation.BLENDS:
        assert result["metrics"]["holdout_2025"]["all"][method]["games"] == 1


def test_blend_uses_point_ratings_and_matchup_audit_detects_reversals():
    common = {"A": 10.0, "B": 0.0, "C": -10.0}
    results = {"A": -10.0, "B": 0.0, "C": 10.0}
    blended = evaluation.blend_ratings(common, results, 0.75)
    assert blended == pytest.approx({"A": -5.0, "B": 0.0, "C": 5.0})
    margins = np.array([[0.0, 10.0, 20.0], [-10.0, 0.0, 10.0], [-20.0, -10.0, 0.0]])
    audit = evaluation.matchup_audit(blended, ["A", "B", "C"], margins)
    assert audit["pair_count"] == 3
    assert audit["disagreements"] == 3
    assert audit["pairs"][0]["higher_team"] == "C"
    assert evaluation.matchup_audit(common, ["A", "B", "C"], margins)["disagreements"] == 0
