from __future__ import annotations

from cfb_prediction.data import (
    normalize_advanced_game_stats,
    normalize_calendar,
    normalize_games,
    normalize_lines,
    normalize_teams,
)


def test_cfbd_records_are_normalized_to_stable_ids() -> None:
    teams = normalize_teams(
        [
            {
                "id": 10,
                "school": "Alpha",
                "abbreviation": "ALP",
                "mascot": "Birds",
                "conference": "Example",
                "classification": "fbs",
                "color": "112233",
                "alternateColor": "ffffff",
                "logos": ["https://example.test/alpha.png"],
            }
        ]
    )
    games = normalize_games(
        [
            {
                "id": 99,
                "season": 2026,
                "week": 1,
                "seasonType": "regular",
                "startDate": "2026-08-29T16:00:00Z",
                "homeId": 10,
                "homeTeam": "Alpha",
                "homeConference": "Example",
                "homeClassification": "fbs",
                "awayId": 20,
                "awayTeam": "Beta",
                "awayConference": "Example",
                "awayClassification": "fbs",
                "neutralSite": False,
                "conferenceGame": True,
                "status": "scheduled",
                "completed": False,
                "homePoints": None,
                "awayPoints": None,
            }
        ]
    )
    calendar = normalize_calendar(
        [
            {
                "season": 2026,
                "week": 1,
                "seasonType": "regular",
                "startDate": "2026-08-29T07:00:00Z",
                "endDate": "2026-09-08T06:59:00Z",
            }
        ]
    )

    assert teams.iloc[0]["team_id"] == 10
    assert teams.iloc[0]["logo"].endswith("alpha.png")
    assert games.iloc[0]["game_id"] == 99
    assert bool(games.iloc[0]["fbs_vs_fbs"]) is True
    assert games.iloc[0]["status"] == "scheduled"
    assert calendar.iloc[0]["week"] == 1


def test_non_fbs_opponent_is_flagged() -> None:
    games = normalize_games(
        [
            {
                "id": 1,
                "season": 2026,
                "week": 1,
                "homeClassification": "fbs",
                "awayClassification": "fcs",
            }
        ]
    )
    assert bool(games.iloc[0]["fbs_vs_fbs"]) is False


def test_advanced_stats_and_market_lines_are_normalized() -> None:
    advanced = normalize_advanced_game_stats(
        [
            {
                "gameId": 7,
                "season": 2025,
                "week": 1,
                "team": "Alpha",
                "opponent": "Beta",
                "offense": {"ppa": 0.2, "successRate": 0.5, "plays": 72},
                "defense": {"ppa": -0.1, "successRate": 0.35},
            }
        ]
    )
    lines = normalize_lines(
        [
            {
                "id": 7,
                "season": 2025,
                "week": 1,
                "lines": [
                    {"provider": "A", "spread": -3.0, "overUnder": 50.0},
                    {"provider": "B", "spread": -4.0, "overUnder": 52.0},
                ],
            }
        ]
    )

    assert advanced.iloc[0]["off_ppa"] == 0.2
    assert advanced.iloc[0]["def_success_rate"] == 0.35
    assert lines.iloc[0]["market_home_margin"] == 3.5
    assert lines.iloc[0]["market_total"] == 51.0
    assert lines.iloc[0]["market_provider_count"] == 2
