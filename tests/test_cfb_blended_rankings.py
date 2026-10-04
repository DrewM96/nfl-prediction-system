from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from test_cfb_features import _historical

from cfb_prediction.blended_rankings import (
    build_blended_cfb_power_ratings,
    common_opponent_ratings,
    freeze_data,
    frozen_frames,
    results_ratings,
)
from cfb_prediction.data import CFBHistoricalData
from cfb_prediction.features import CFB_ELO_FEATURES


def test_published_blend_uses_frozen_results_and_survives_no_remaining_schedule():
    data = _historical()
    data.games["home_conference"] = "SEC"
    data.games["away_conference"] = "ACC"
    args = dict(
        margin_estimator=LinearModel(),
        preseason_estimator=LinearModel(),
        feature_names=CFB_ELO_FEATURES,
        created_at=datetime(2025, 9, 5, tzinfo=UTC),
        prediction_season=2025,
        forecast_week=2,
        model_hash="test",
        input_coverage={},
    )
    before = build_blended_cfb_power_ratings(data, **args)
    data.games.loc[data.games.week.eq(2), "home_points"] = 99
    data.advanced.loc[data.advanced.game_id.eq(2), "off_ppa"] = 100
    assert build_blended_cfb_power_ratings(data, **args) == before
    assert before["kind"] == "blended_common_opponent_results"
    assert before["method"]["results_weight"] == 0.25
    assert before["method"]["margin_cap"] is None
    assert before["game_count"] == 1
    assert before["team_count"] == 2
    assert sum(r["rating"] for r in before["ratings"]) == pytest.approx(0)
    for row in before["ratings"]:
        assert row["rating"] == pytest.approx(
            0.75 * row["common_opponent_rating"] + 0.25 * row["results_rating"]
        )
    after = build_blended_cfb_power_ratings(
        data, **{**args, "forecast_week": 3, "created_at": datetime(2025, 9, 10, tzinfo=UTC)}
    )
    assert after["game_count"] == 2
    assert all(r["scheduled_games"] == 0 for r in after["ratings"])
    assert after["team_count"] == 2


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
