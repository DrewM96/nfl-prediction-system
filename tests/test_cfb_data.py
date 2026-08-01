from __future__ import annotations

from cfb_prediction.data import normalize_calendar, normalize_games, normalize_teams


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
