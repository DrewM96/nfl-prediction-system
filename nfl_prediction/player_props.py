"""Pure joins of existing projections, depth charts, and main sportsbook lines."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from statistics import median
from typing import Any

from .current_market import stale_at
from .odds import _game_kickoff, parse_timestamp
from .owls import OwlsError, match_event, name_key, number, team_aliases, timestamp

PROP_MODELS = {
    "passing_yards": "qb",
    "rushing_yards": "rb",
    "receiving_yards": "wr",
    "receptions": "wr",
}
PROP_TITLES = {
    "passing_yards": "Passing yards",
    "rushing_yards": "Rushing yards",
    "receiving_yards": "Receiving yards",
    "receptions": "Receptions",
}
PROP_MAX_AGE = 3600
DEPTH_MAX_AGE = 172800


def player_name_key(value: str) -> str:
    # Remove only suffixes, never collapse first names to initials.
    value = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", value.strip(), flags=re.I)
    return name_key(value)


def prop_key(row: dict[str, Any]) -> tuple:
    return row["event_id"], row["sportsbook"], row["player_key"], row["category"]


def parse_props(payload: dict[str, Any], slate: list[dict[str, Any]], captured_at: str) -> dict:
    if payload.get("success") is False or not isinstance(payload.get("data"), list):
        raise OwlsError("Owls props: malformed board")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict) or not isinstance(meta.get("books", {}), dict):
        raise OwlsError("Owls props: malformed metadata")
    rows: dict[tuple, dict] = {}
    conflicts = set()
    diagnostics: Counter = Counter()
    aliases = team_aliases("nfl", slate)
    for event in payload["data"]:
        if not isinstance(event, dict) or not isinstance(event.get("books"), list):
            raise OwlsError("Owls props: malformed event")
        mapped = {
            "home_team": event.get("homeTeam"),
            "away_team": event.get("awayTeam"),
            "commence_time": event.get("commenceTime"),
        }
        game, reason = match_event(mapped, "nfl", slate)
        if game is None:
            diagnostics[reason] += 1
            continue
        if event.get("isLive") or not isinstance(event.get("gameId"), str) or not event["gameId"]:
            diagnostics["live_or_missing_event_id"] += 1
            continue
        for book in event["books"]:
            if not isinstance(book, dict) or not isinstance(book.get("props"), list):
                raise OwlsError("Owls props: malformed sportsbook")
            if not isinstance(book.get("key"), str) or book["key"] not in {
                "pinnacle",
                "fanduel",
                "draftkings",
                "caesars",
                "betmgm",
                "bet365",
            }:
                continue
            book_meta = meta.get("books", {}).get(book["key"], {})
            if not isinstance(book_meta, dict):
                raise OwlsError("Owls props: malformed sportsbook metadata")
            for prop in book["props"]:
                if not isinstance(prop, dict):
                    diagnostics["malformed_prop"] += 1
                    continue
                if not isinstance(prop.get("category"), str) or prop["category"] not in PROP_MODELS:
                    continue
                line, over, under = (
                    number(prop.get(k)) for k in ("line", "overPrice", "underPrice")
                )
                structure = prop.get("marketStructure")
                if (
                    line is None
                    or line < 0
                    or over is None
                    or under is None
                    or abs(over) < 100
                    or abs(under) < 100
                    or prop.get("isMain") is False
                    or prop.get("isAlternate") is True
                    or structure not in (None, "over_under")
                    or (structure is None and line % 1 != 0.5)
                ):
                    diagnostics["unsupported_or_ambiguous_line"] += 1
                    continue
                player = prop.get("playerName")
                if not isinstance(player, str) or not player_name_key(player):
                    diagnostics["missing_player"] += 1
                    continue
                raw_team = prop.get("team")
                team = aliases.get(name_key(str(raw_team))) if raw_team else None
                if raw_team and team not in {game["home_team"], game["away_team"]}:
                    diagnostics["player_team_mismatch"] += 1
                    continue
                row = {
                    "event_id": str(event["gameId"]),
                    "game_id": str(game["game_id"]),
                    "commence_time": timestamp(mapped["commence_time"]),
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "player_name": player,
                    "player_key": player_name_key(player),
                    "team": team,
                    "category": prop["category"],
                    "sportsbook": str(book["key"]),
                    "line": line,
                    "over_price": over,
                    "under_price": under,
                    "source_timestamp": timestamp(prop.get("lastUpdate") or book.get("lastUpdate")),
                    "captured_at": captured_at,
                    "provider_stale": book_meta.get("status") not in (None, "ok", "fresh"),
                }
                key = prop_key(row)
                if key in rows and any(
                    rows[key][field] != row[field]
                    for field in ("line", "over_price", "under_price", "team")
                ):
                    conflicts.add(key)
                    diagnostics["conflicting_main_lines"] += 1
                else:
                    rows[key] = row
    return {
        "rows": [row for key, row in rows.items() if key not in conflicts],
        "diagnostics": dict(diagnostics),
    }


def projection_rows(state: dict, models: dict, manifest: dict) -> list[dict]:
    """Called once per release by the UI cache, never by the odds poller."""
    import pandas as pd

    result = []
    season = int(manifest["prediction_season"])
    for category, group in PROP_MODELS.items():
        for player_id, player in state.get(group, {}).items():
            if player.get("prediction_season") != season or player.get("roster_season") != season:
                continue
            games = [
                g
                for g in state.get("schedule", [])
                if {g["home_team"], g["away_team"]} == {player.get("team"), player.get("opponent")}
            ]
            if len(games) != 1:
                continue
            game = games[0]
            start = _game_kickoff(game)
            if start is None:
                continue
            # Exactly the existing PredictionService distribution and mean clamp.
            distribution = models[category].distribution(pd.DataFrame([player]))[0]
            value = number(distribution.get("mean"))
            if value is None:
                continue
            result.append(
                {
                    "player_id": str(player_id),
                    "player_name": player["player_name"],
                    "team": player["team"],
                    "opponent": player["opponent"],
                    "game_id": str(game["game_id"]),
                    "commence_time": start.isoformat(),
                    "category": category,
                    "projection": max(value, 0.0),
                }
            )
    return result


def comparisons(
    projections: list[dict],
    board: dict,
    depth: dict,
    category: str,
    *,
    book: str = "Consensus",
    include_stale: bool = False,
    now: datetime | None = None,
) -> list[dict]:
    now = now or datetime.now(UTC)
    # Unknown or outdated depth charts never silently become starter assignments.
    starters = {
        (p["team"], p["player_id"]): p
        for p in depth.get("players", [])
        if p.get("starter") and not stale_at(p.get("source_timestamp"), now, max_age=DEPTH_MAX_AGE)
    }
    candidates: dict[tuple, list] = defaultdict(list)
    for p in projections:
        if p["category"] != category or parse_timestamp(p["commence_time"]) <= now:
            continue
        starter = starters.get((p["team"], p["player_id"]))
        if starter:
            candidates[(p["game_id"], player_name_key(starter["player_name"]))].append((p, starter))
    grouped: dict[tuple, list] = defaultdict(list)
    for quote in board.get("rows", []):
        if quote["category"] != category or (book != "Consensus" and book != quote["sportsbook"]):
            continue
        matches = candidates.get((quote["game_id"], quote["player_key"]), [])
        matches = [
            (p, s)
            for p, s in matches
            if (not quote.get("team") or p["team"] == quote["team"])
            and p["commence_time"] == quote["commence_time"]
        ]
        if len(matches) != 1:
            continue
        stale = bool(
            board.get("error")
            or quote.get("provider_stale")
            or quote.get("unavailable")
            or stale_at(quote.get("source_timestamp"), now, max_age=PROP_MAX_AGE)
            or stale_at(quote.get("captured_at"), now, max_age=PROP_MAX_AGE)
        )
        if stale and not include_stale:
            continue
        projection, starter = matches[0]
        grouped[(projection["game_id"], projection["player_id"])].append(
            ({**quote, "stale": stale}, projection, starter)
        )
    result = []
    for entries in grouped.values():
        quotes = [q for q, _, _ in entries]
        projection, starter = entries[0][1:]
        line = median(q["line"] for q in quotes)
        difference = projection["projection"] - line
        result.append(
            {
                **projection,
                "player_name": starter["player_name"],
                "position": starter["position"],
                "line": line,
                "difference": difference,
                "lean": "Over" if difference > 0 else "Under" if difference < 0 else "Even",
                "books": len(quotes),
                "quotes": quotes,
                "stale": any(q["stale"] for q in quotes),
                "source_timestamp": (
                    min(q["source_timestamp"] for q in quotes)
                    if all(q.get("source_timestamp") for q in quotes)
                    else None
                ),
                "depth_timestamp": starter["source_timestamp"],
            }
        )
    return sorted(result, key=lambda r: (-abs(r["difference"]), r["player_name"], r["game_id"]))
