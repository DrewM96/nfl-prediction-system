"""Temporary, read-only provider diagnostics on an identical GRIDLINE slate."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import Any

from .current_market import current_context
from .odds import OddsApiClient, OddsApiError, _game_kickoff
from .owls import SPORT_KEYS, OwlsClient, OwlsError, parse_odds


def compare_boards(
    owls: dict[str, Any],
    legacy: dict[str, Any],
    slate: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    left = {g["game_id"]: g for g in owls["games"]}
    right = {g["game_id"]: g for g in legacy["games"]}
    rows = []
    for game in slate:
        gid = str(game["game_id"])
        a, b = left.get(gid, {}), right.get(gid, {})
        sa, sb = (
            (a.get("spread") or {}).get("home_spread"),
            (b.get("spread") or {}).get("home_spread"),
        )
        books_a, books_b = set(a.get("books", {})), set(b.get("books", {}))
        common = books_a & books_b
        common_spreads_a = [
            a["books"][key]["spread"]
            for key in common
            if a["books"][key].get("spread") is not None
            and b["books"][key].get("spread") is not None
        ]
        common_spreads_b = [
            b["books"][key]["spread"]
            for key in common
            if a["books"][key].get("spread") is not None
            and b["books"][key].get("spread") is not None
        ]
        kickoff = _game_kickoff(game)
        rows.append(
            {
                "game_id": gid,
                "pregame": kickoff is not None and kickoff > now,
                "owls_event_id": a.get("event_id"),
                "legacy_event_id": b.get("event_id"),
                "owls_spread": sa,
                "legacy_spread": sb,
                "spread_difference": round(sa - sb, 3)
                if sa is not None and sb is not None
                else None,
                "owls_books": sorted(books_a),
                "legacy_books": sorted(books_b),
                "common_books": sorted(books_a & books_b),
                "common_book_spread_difference": round(
                    statistics.median(common_spreads_a) - statistics.median(common_spreads_b), 3
                )
                if common_spreads_a
                else None,
                "owls_timestamp": a.get("source_timestamp"),
                "legacy_timestamp": b.get("source_timestamp"),
                "owls_freshness": current_context(game, owls, now=now)["status"],
                "legacy_freshness": current_context(game, legacy, now=now)["status"],
                "normalization": {
                    "owls": [a.get("away_team_name"), a.get("home_team_name")],
                    "legacy": [b.get("away_team_name"), b.get("home_team_name")],
                },
            }
        )
    return {
        "checked_at": now.isoformat(),
        "slate_games": len(slate),
        "owls_coverage": len(left),
        "legacy_coverage": len(right),
        "pregame_slate_games": sum(row["pregame"] for row in rows),
        "owls_pregame_coverage": sum(
            row["pregame"] and row["owls_spread"] is not None for row in rows
        ),
        "legacy_pregame_coverage": sum(
            row["pregame"] and row["legacy_spread"] is not None for row in rows
        ),
        "owls_unmatched": owls.get("diagnostics", []),
        "legacy_unmatched": legacy.get("diagnostics", []),
        "missing_from_owls": sorted(
            str(g["game_id"]) for g in slate if str(g["game_id"]) not in left
        ),
        "missing_from_legacy": sorted(
            str(g["game_id"]) for g in slate if str(g["game_id"]) not in right
        ),
        "games": rows,
        "retirement_verified": False,
        "note": "Diagnostics only. Review multiple fresh active slates and common-book differences before retirement.",
    }


def live_parity(sport: str, slate: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        raw = OwlsClient().get(sport, "odds")
        owls = parse_odds(raw, sport, slate, datetime.now(UTC).isoformat())
        fetch = OddsApiClient().current_odds(sport_key=SPORT_KEYS[sport])
        legacy = parse_odds(
            {"data": {"legacy": fetch.payload}},
            sport,
            slate,
            fetch.captured_at,
            provider="The Odds API",
        )
        return compare_boards(owls, legacy, slate)
    except (OwlsError, OddsApiError) as exc:
        return {
            "sport": sport,
            "status": "unverified",
            "reason": str(exc),
            "retirement_verified": False,
        }
