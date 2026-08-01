from __future__ import annotations

import pandas as pd

from cfb_prediction.data import CFBHistoricalData
from cfb_prediction.features import CFB_FULL_FEATURES, build_point_in_time_features


def _historical(first_home_points: int = 40) -> CFBHistoricalData:
    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "season": 2025,
                "week": 1,
                "start_date": pd.Timestamp("2025-08-30T16:00:00Z"),
                "home_id": 10,
                "home_team": "Alpha",
                "home_classification": "fbs",
                "away_id": 20,
                "away_team": "Beta",
                "away_classification": "fbs",
                "neutral_site": False,
                "conference_game": False,
                "completed": True,
                "home_points": first_home_points,
                "away_points": 10,
                "fbs_vs_fbs": True,
            },
            {
                "game_id": 2,
                "season": 2025,
                "week": 2,
                "start_date": pd.Timestamp("2025-09-06T16:00:00Z"),
                "home_id": 10,
                "home_team": "Alpha",
                "home_classification": "fbs",
                "away_id": 20,
                "away_team": "Beta",
                "away_classification": "fbs",
                "neutral_site": False,
                "conference_game": True,
                "completed": True,
                "home_points": 24,
                "away_points": 21,
                "fbs_vs_fbs": True,
            },
        ]
    )
    advanced_rows = []
    for game_id in (1, 2):
        for team, opponent, ppa in (("Alpha", "Beta", 0.3), ("Beta", "Alpha", -0.1)):
            advanced_rows.append(
                {
                    "game_id": game_id,
                    "season": 2025,
                    "week": game_id,
                    "team": team,
                    "opponent": opponent,
                    "off_ppa": ppa,
                    "def_ppa": -ppa,
                    "off_success_rate": 0.5,
                    "def_success_rate": 0.4,
                    "off_explosiveness": 1.3,
                    "def_explosiveness": 1.1,
                    "off_plays": 70,
                }
            )
    return CFBHistoricalData(
        games=games,
        advanced=pd.DataFrame(advanced_rows),
        returning=pd.DataFrame(
            [
                {
                    "season": 2025,
                    "team": team,
                    "returning_ppa": value,
                    "returning_passing_ppa": value,
                    "returning_rushing_ppa": value,
                    "returning_receiving_ppa": value,
                }
                for team, value in (("Alpha", 0.8), ("Beta", 0.4))
            ]
        ),
        portal=pd.DataFrame(
            [
                {
                    "season": 2025,
                    "origin": "Beta",
                    "destination": "Alpha",
                    "rating": 0.9,
                    "transfer_date": pd.Timestamp("2025-05-01T12:00:00Z"),
                },
                {
                    "season": 2025,
                    "origin": "Alpha",
                    "destination": "Beta",
                    "rating": 100.0,
                    "transfer_date": pd.Timestamp("2025-12-01T12:00:00Z"),
                },
            ]
        ),
        talent=pd.DataFrame(
            [
                {"season": 2025, "team": "Alpha", "talent": 800},
                {"season": 2025, "team": "Beta", "talent": 400},
            ]
        ),
        recruiting=pd.DataFrame(
            [
                {"season": 2025, "team": "Alpha", "recruiting_points": 250},
                {"season": 2025, "team": "Beta", "recruiting_points": 150},
            ]
        ),
        lines=pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "market_home_margin": 6.5,
                    "market_total": 52.5,
                    "market_provider_count": 2,
                }
            ]
        ),
    )


def test_features_are_created_before_each_result_updates_state() -> None:
    original = build_point_in_time_features(_historical())
    changed_result = build_point_in_time_features(_historical(first_home_points=3))

    assert len(original) == 2
    assert original.iloc[0]["home_points_for_l6"] == 27.0
    assert original.iloc[0][CFB_FULL_FEATURES].equals(changed_result.iloc[0][CFB_FULL_FEATURES])
    assert original.iloc[1]["home_points_for_l6"] > changed_result.iloc[1]["home_points_for_l6"]


def test_features_include_opponent_and_preseason_context() -> None:
    games = build_point_in_time_features(_historical())
    first = games.iloc[0]

    assert first["elo_expected_margin"] > 0
    assert first["home_talent_z"] > first["away_talent_z"]
    assert first["home_portal_net_rating_z"] > first["away_portal_net_rating_z"]
    assert first["market_home_margin"] == 6.5
