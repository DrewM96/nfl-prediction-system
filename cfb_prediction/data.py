from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CFBFoundationData:
    teams: pd.DataFrame
    games: pd.DataFrame
    calendar: pd.DataFrame


@dataclass(frozen=True)
class CFBHistoricalData:
    games: pd.DataFrame
    advanced: pd.DataFrame
    returning: pd.DataFrame
    portal: pd.DataFrame
    talent: pd.DataFrame
    recruiting: pd.DataFrame
    lines: pd.DataFrame


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
        "status",
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
            "status": _column(frame, "status", ""),
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


def normalize_advanced_game_stats(records: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records if isinstance(records, list) else []:
        offense = record.get("offense") or {}
        defense = record.get("defense") or {}
        rows.append(
            {
                "game_id": record.get("gameId"),
                "season": record.get("season"),
                "week": record.get("week"),
                "team": record.get("team"),
                "opponent": record.get("opponent"),
                "off_ppa": offense.get("ppa"),
                "off_success_rate": offense.get("successRate"),
                "off_explosiveness": offense.get("explosiveness"),
                "off_plays": offense.get("plays"),
                "off_drives": offense.get("drives"),
                "def_ppa": defense.get("ppa"),
                "def_success_rate": defense.get("successRate"),
                "def_explosiveness": defense.get("explosiveness"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
                "off_ppa",
                "off_success_rate",
                "off_explosiveness",
                "off_plays",
                "off_drives",
                "def_ppa",
                "def_success_rate",
                "def_explosiveness",
            ]
        )
    for column in ("game_id", "season", "week"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    numeric = [
        column
        for column in frame
        if column not in {"game_id", "season", "week", "team", "opponent"}
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["game_id", "team"]).drop_duplicates(
        ["game_id", "team"], keep="last"
    )


def normalize_returning_production(records: Any) -> pd.DataFrame:
    frame = _frame(records)
    columns = [
        "season",
        "team",
        "returning_ppa",
        "returning_passing_ppa",
        "returning_rushing_ppa",
        "returning_receiving_ppa",
        "returning_usage",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(
        {
            "season": pd.to_numeric(_column(frame, "season", None), errors="coerce").astype(
                "Int64"
            ),
            "team": _column(frame, "team", ""),
            "returning_ppa": pd.to_numeric(_column(frame, "percentPPA", None), errors="coerce"),
            "returning_passing_ppa": pd.to_numeric(
                _column(frame, "percentPassingPPA", None), errors="coerce"
            ),
            "returning_rushing_ppa": pd.to_numeric(
                _column(frame, "percentRushingPPA", None), errors="coerce"
            ),
            "returning_receiving_ppa": pd.to_numeric(
                _column(frame, "percentReceivingPPA", None), errors="coerce"
            ),
            "returning_usage": pd.to_numeric(_column(frame, "usage", None), errors="coerce"),
        }
    )
    return result.dropna(subset=["season", "team"]).drop_duplicates(["season", "team"], keep="last")


def normalize_portal(records: Any) -> pd.DataFrame:
    frame = _frame(records)
    columns = ["season", "origin", "destination", "rating", "stars", "transfer_date"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(
        {
            "season": pd.to_numeric(_column(frame, "season", None), errors="coerce").astype(
                "Int64"
            ),
            "origin": _column(frame, "origin", ""),
            "destination": _column(frame, "destination", ""),
            "rating": pd.to_numeric(_column(frame, "rating", None), errors="coerce"),
            "stars": pd.to_numeric(_column(frame, "stars", None), errors="coerce"),
            "transfer_date": pd.to_datetime(
                _column(frame, "transferDate", None), errors="coerce", utc=True
            ),
        }
    )
    return result.dropna(subset=["season"]).reset_index(drop=True)


def normalize_talent(records: Any) -> pd.DataFrame:
    frame = _frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["season", "team", "talent"])
    return pd.DataFrame(
        {
            "season": pd.to_numeric(_column(frame, "year", None), errors="coerce").astype("Int64"),
            "team": _column(frame, "team", ""),
            "talent": pd.to_numeric(_column(frame, "talent", None), errors="coerce"),
        }
    ).dropna(subset=["season", "team"])


def normalize_recruiting(records: Any) -> pd.DataFrame:
    frame = _frame(records)
    if frame.empty:
        return pd.DataFrame(columns=["season", "team", "recruiting_points", "recruiting_rank"])
    return pd.DataFrame(
        {
            "season": pd.to_numeric(_column(frame, "year", None), errors="coerce").astype("Int64"),
            "team": _column(frame, "team", ""),
            "recruiting_points": pd.to_numeric(_column(frame, "points", None), errors="coerce"),
            "recruiting_rank": pd.to_numeric(_column(frame, "rank", None), errors="coerce"),
        }
    ).dropna(subset=["season", "team"])


def normalize_lines(records: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in records if isinstance(records, list) else []:
        providers = game.get("lines") or []
        spreads = pd.to_numeric(
            pd.Series([line.get("spread") for line in providers]), errors="coerce"
        ).dropna()
        totals = pd.to_numeric(
            pd.Series([line.get("overUnder") for line in providers]), errors="coerce"
        ).dropna()
        rows.append(
            {
                "game_id": game.get("id"),
                "season": game.get("season"),
                "week": game.get("week"),
                "market_home_margin": -float(spreads.median()) if not spreads.empty else None,
                "market_total": float(totals.median()) if not totals.empty else None,
                "market_provider_count": len(providers),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "market_home_margin",
                "market_total",
                "market_provider_count",
            ]
        )
    for column in ("game_id", "season", "week", "market_provider_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame.dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")
