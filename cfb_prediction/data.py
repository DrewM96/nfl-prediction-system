from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CFBFoundationData:
    teams: pd.DataFrame
    games: pd.DataFrame
    calendar: pd.DataFrame


def _frame(records: Any) -> pd.DataFrame:
    return pd.DataFrame(records if isinstance(records, list) else [])


def _column(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series([default for _ in frame.index], index=frame.index)


def normalize_teams(records: Any) -> pd.DataFrame:
    frame = _frame(records)
    columns = [
        "team_id",
        "school",
        "abbreviation",
        "mascot",
        "conference",
        "classification",
        "color",
        "alternate_color",
        "logo",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(
        {
            "team_id": pd.to_numeric(_column(frame, "id", None), errors="coerce").astype("Int64"),
            "school": _column(frame, "school", ""),
            "abbreviation": _column(frame, "abbreviation", ""),
            "mascot": _column(frame, "mascot", ""),
            "conference": _column(frame, "conference", ""),
            "classification": _column(frame, "classification", ""),
            "color": _column(frame, "color", ""),
            "alternate_color": _column(frame, "alternateColor", ""),
            "logo": _column(frame, "logos", []).map(
                lambda logos: logos[0] if isinstance(logos, list) and logos else ""
            ),
        }
    )
    return result.dropna(subset=["team_id"]).drop_duplicates("team_id").reset_index(drop=True)


def normalize_games(records: Any) -> pd.DataFrame:
    frame = _frame(records)
    columns = [
        "game_id",
        "season",
        "week",
        "season_type",
        "start_date",
        "home_id",
        "home_team",
        "home_conference",
        "home_classification",
        "away_id",
        "away_team",
        "away_conference",
        "away_classification",
        "neutral_site",
        "conference_game",
        "completed",
        "home_points",
        "away_points",
        "fbs_vs_fbs",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(
        {
            "game_id": pd.to_numeric(_column(frame, "id", None), errors="coerce").astype("Int64"),
            "season": pd.to_numeric(_column(frame, "season", None), errors="coerce").astype(
                "Int64"
            ),
            "week": pd.to_numeric(_column(frame, "week", None), errors="coerce").astype("Int64"),
            "season_type": _column(frame, "seasonType", ""),
            "start_date": pd.to_datetime(
                _column(frame, "startDate", None), errors="coerce", utc=True
            ),
            "home_id": pd.to_numeric(_column(frame, "homeId", None), errors="coerce").astype(
                "Int64"
            ),
            "home_team": _column(frame, "homeTeam", ""),
            "home_conference": _column(frame, "homeConference", ""),
            "home_classification": _column(frame, "homeClassification", ""),
            "away_id": pd.to_numeric(_column(frame, "awayId", None), errors="coerce").astype(
                "Int64"
            ),
            "away_team": _column(frame, "awayTeam", ""),
            "away_conference": _column(frame, "awayConference", ""),
            "away_classification": _column(frame, "awayClassification", ""),
            "neutral_site": _column(frame, "neutralSite", False),
            "conference_game": _column(frame, "conferenceGame", False),
            "completed": _column(frame, "completed", False),
            "home_points": pd.to_numeric(_column(frame, "homePoints", None), errors="coerce"),
            "away_points": pd.to_numeric(_column(frame, "awayPoints", None), errors="coerce"),
        }
    )
    result["fbs_vs_fbs"] = result["home_classification"].eq("fbs") & result[
        "away_classification"
    ].eq("fbs")
    return result.dropna(subset=["game_id"]).drop_duplicates("game_id").reset_index(drop=True)


def normalize_calendar(records: Any) -> pd.DataFrame:
    frame = _frame(records)
    columns = ["season", "week", "season_type", "start_date", "end_date"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "season": pd.to_numeric(_column(frame, "season", None), errors="coerce").astype(
                "Int64"
            ),
            "week": pd.to_numeric(_column(frame, "week", None), errors="coerce").astype("Int64"),
            "season_type": _column(frame, "seasonType", ""),
            "start_date": pd.to_datetime(
                _column(frame, "startDate", None), errors="coerce", utc=True
            ),
            "end_date": pd.to_datetime(_column(frame, "endDate", None), errors="coerce", utc=True),
        }
    ).sort_values(["season", "week"], ignore_index=True)
