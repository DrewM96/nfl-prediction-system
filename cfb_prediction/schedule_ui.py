"""Read-only filters for the weekly college football schedule."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

TIME_SLOTS = ("Early", "Afternoon", "Primetime", "Late", "Time TBD")


def kickoff_et(game: dict[str, Any]) -> datetime | None:
    try:
        kickoff = datetime.fromisoformat(str(game.get("start_date")))
    except (TypeError, ValueError):
        return None
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=UTC)
    return kickoff.astimezone(ZoneInfo("America/New_York"))


def kickoff_day(game: dict[str, Any]) -> str:
    kickoff = kickoff_et(game)
    return f"{kickoff.strftime('%a')} {kickoff.month}/{kickoff.day}" if kickoff else "Time TBD"


def time_slot(game: dict[str, Any]) -> str:
    kickoff = kickoff_et(game)
    if kickoff is None:
        return "Time TBD"
    if kickoff.hour < 15:
        return "Early"
    if kickoff.hour < 19:
        return "Afternoon"
    if kickoff.hour < 22:
        return "Primetime"
    return "Late"


def game_conferences(game: dict[str, Any], conferences: dict[str, str]) -> set[str]:
    return {
        str(conference)
        for side in ("home", "away")
        if (conference := game.get(f"{side}_conference") or conferences.get(game[f"{side}_team"]))
    }


def matches_kickoff_filter(game: dict[str, Any], slot: str, completed_ids: set[str]) -> bool:
    if slot == "All times":
        return True
    if slot == "Completed":
        return str(game["game_id"]) in completed_ids
    if slot in {"Weekday", "Saturday"}:
        kickoff = kickoff_et(game)
        return kickoff is not None and (
            kickoff.weekday() < 5 if slot == "Weekday" else kickoff.weekday() == 5
        )
    return time_slot(game) == slot


def filter_schedule(
    games: list[dict[str, Any]],
    conferences: dict[str, str],
    *,
    conference: str = "All conferences",
    day: str = "All days",
    slot: str = "All times",
    search: str = "",
    completed_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    query = search.strip().casefold()
    return [
        game
        for game in games
        if (conference == "All conferences" or conference in game_conferences(game, conferences))
        and (day == "All days" or kickoff_day(game) == day)
        and matches_kickoff_filter(game, slot, completed_ids or set())
        and (
            not query
            or any(query in str(game[f"{side}_team"]).casefold() for side in ("home", "away"))
        )
    ]
