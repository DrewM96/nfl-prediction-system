from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from nfl_prediction.rankings import build_market_power_ratings, build_model_power_ratings


def test_model_power_ratings_recover_additive_strength_and_home_field() -> None:
    strengths = {"A": 4.0, "B": 1.0, "C": -1.0, "D": -4.0}

    def predict_margin(away: str, home: str, neutral_site: bool) -> float:
        return strengths[home] - strengths[away] + (0.0 if neutral_site else 2.25)

    result = build_model_power_ratings(
        list(strengths),
        predict_margin=predict_margin,
        prediction_week=7,
    )

    assert result is not None
    assert result["prediction_week"] == 7
    assert result["matchup_count"] == 6
    assert result["home_field_points"] == pytest.approx(2.25)
    assert result["reconstruction_mae"] == pytest.approx(0.0, abs=1e-10)
    assert result["directional_asymmetry_mae"] == pytest.approx(0.0, abs=1e-10)
    ratings = {row["team"]: row["rating"] for row in result["ratings"]}
    assert ratings == pytest.approx(strengths)


def test_model_power_ratings_antisymmetrize_directional_noise() -> None:
    strengths = {"A": 3.0, "B": 0.0, "C": -3.0}

    def predict_margin(away: str, home: str, neutral_site: bool) -> float:
        directional_bias = 0.4
        return (
            strengths[home]
            - strengths[away]
            + directional_bias
            + (0.0 if neutral_site else 2.0)
        )

    result = build_model_power_ratings(list(strengths), predict_margin=predict_margin)

    assert result is not None
    ratings = {row["team"]: row["rating"] for row in result["ratings"]}
    assert ratings == pytest.approx(strengths)
    assert result["directional_asymmetry_mae"] == pytest.approx(0.4)


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


def test_published_market_snapshot_is_valid() -> None:
    snapshot = json.loads(Path("market_consensus.json").read_text(encoding="utf-8"))

    assert snapshot.get("games")
    assert snapshot.get("snapshot_at")

    result = build_market_power_ratings(snapshot)
    if result is None:
        return

    assert result["team_count"] >= 2
    assert len(result["ratings"]) == result["team_count"]
    assert result["game_count"] + result["excluded_single_book_games"] <= len(snapshot["games"])
    assert result["median_book_count"] >= 2
    assert result["line_fit_mae"] < 1.0
    assert min(row["games"] for row in result["ratings"]) > 0


def test_market_ratings_remove_home_field_for_known_neutral_matchup() -> None:
    strengths = {"A": 4.0, "B": 1.0, "C": -1.0, "D": -4.0}
    games = []
    for first, second in itertools.combinations(strengths, 2):
        for home, away in ((first, second), (second, first)):
            home_field = 0.0 if (away, home) == ("B", "A") else 2.0
            games.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "spread": {
                        "market_home_margin": strengths[home] - strengths[away] + home_field,
                        "book_count": 10,
                    },
                }
            )

    result = build_market_power_ratings(
        {"games": games},
        neutral_matchups={("B", "A")},
    )

    assert result is not None
    assert result["home_field_points"] == pytest.approx(2.0)
    ratings = {row["team"]: row["rating"] for row in result["ratings"]}
    assert ratings == pytest.approx(strengths)
