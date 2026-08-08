from __future__ import annotations

import itertools

import pytest

from nfl_prediction.preseason import apply_preseason_calibration


def _snapshot() -> dict:
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
    return {"snapshot_at": "2026-08-01T12:00:00+00:00", "games": games}


def _prediction(week: int) -> dict:
    return {
        "home_team": "A",
        "away_team": "B",
        "week": week,
        "predicted_home_margin": 1.0,
        "total": 44.0,
        "home_score": 22.5,
        "away_score": 21.5,
        "home_win_probability": 0.53,
        "margin_std": 10.0,
        "neutral_site": True,
        "features": {"home_field": 0.0},
    }


def test_week_one_uses_neutral_field_preseason_strength_and_preserves_football_model() -> None:
    result = apply_preseason_calibration([_prediction(1)], _snapshot())[0]

    assert result["predicted_home_margin"] == pytest.approx(3.0)
    assert result["home_score"] == pytest.approx(23.5)
    assert result["away_score"] == pytest.approx(20.5)
    assert result["football_only"]["home_margin"] == pytest.approx(1.0)
    assert result["preseason_calibration"]["weight"] == 1.0
    assert result["preseason_calibration"]["neutral_site"] is True


def test_preseason_strength_tapers_through_week_four() -> None:
    results = apply_preseason_calibration(
        [_prediction(week) for week in (1, 2, 3, 4, 5)],
        _snapshot(),
    )

    assert [game["predicted_home_margin"] for game in results] == [3.0, 2.5, 2.0, 1.5, 1.0]
    assert [game["preseason_calibration"]["weight"] for game in results] == [
        1.0,
        0.75,
        0.5,
        0.25,
        0.0,
    ]


def test_calibration_is_idempotent_for_the_same_market_snapshot() -> None:
    first = apply_preseason_calibration([_prediction(1)], _snapshot())
    second = apply_preseason_calibration(first, _snapshot())

    assert second == first
