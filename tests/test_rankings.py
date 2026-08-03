from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from nfl_prediction.rankings import build_market_power_ratings


def test_market_ratings_recover_neutral_strength_and_home_field() -> None:
    strengths = {"A": 4.0, "B": 1.0, "C": -1.0, "D": -4.0}
    games = []
    for first, second in itertools.combinations(strengths, 2):
        for home, away in ((first, second), (second, first)):
            games.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "spread": {
                        "market_home_margin": strengths[home] - strengths[away] + 2.0,
                        "book_count": 10,
                        "line_iqr": 0.0,
                    },
                }
            )
    result = build_market_power_ratings(
        {"snapshot_at": "2026-08-02T20:11:12+00:00", "games": games}
    )

    assert result is not None
    assert result["home_field_points"] == pytest.approx(2.0)
    assert result["line_fit_mae"] == pytest.approx(0.0, abs=1e-10)
    ratings = {row["team"]: row["rating"] for row in result["ratings"]}
    assert ratings == pytest.approx(strengths)


def test_published_market_snapshot_produces_full_league_ranking() -> None:
    snapshot = json.loads(Path("market_consensus.json").read_text(encoding="utf-8"))
    result = build_market_power_ratings(snapshot)

    assert result is not None
    assert result["game_count"] == 261
    assert result["excluded_single_book_games"] == 11
    assert result["team_count"] == 32
    assert result["median_book_count"] >= 2
    assert result["line_fit_mae"] < 1.0
    assert min(row["games"] for row in result["ratings"]) >= 15
