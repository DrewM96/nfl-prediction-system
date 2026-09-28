"""Owls normalized v1 adapter. Market context only; no model dependencies."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import PROJECT_ROOT
from .io import read_json
from .odds import TEAM_NAME_TO_CODE, _game_kickoff, _market_summary, parse_timestamp

SPORT_KEYS = {"nfl": "americanfootball_nfl", "ncaaf": "americanfootball_ncaaf"}


class OwlsError(RuntimeError):
    """Redacted error; never include request headers, URLs or response bodies."""

    def __init__(self, message: str, *, retry_after: int = 300):
        super().__init__(message)
        self.retry_after = retry_after


def timestamp(value: Any) -> str | None:
    try:
        return parse_timestamp(value).isoformat() if isinstance(value, str) else None
    except (ValueError, TypeError):
        return None


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def retry_seconds(value: str | None) -> int:
    try:
        return max(300, int(value or "300"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value or "")
            return max(300, int((parsed - datetime.now(UTC)).total_seconds())) if parsed else 300
        except (TypeError, ValueError):
            return 300


class OwlsClient:
    def __init__(self) -> None:
        self.quota_retry_after = 0

    def get(self, sport: str, endpoint: str) -> dict[str, Any]:
        if sport not in SPORT_KEYS or endpoint not in {"odds", "splits"}:
            raise ValueError("Unsupported Owls sport or endpoint")
        key = os.environ.get("OWLS_INSIGHT_API_KEY", "").strip()
        if not key:
            raise OwlsError("OWLS_INSIGHT_API_KEY is not configured", retry_after=3600)
        suffix = "?exclude_exchanges=true" if endpoint == "odds" else ""
        request = Request(
            f"https://api.owlsinsight.com/api/v1/{sport}/{endpoint}{suffix}",
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "User-Agent": "GRIDLINE/4.0 (market context)",
            },
        )
        try:
            with urlopen(request, timeout=20) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
                remaining = number(response.headers.get("X-RateLimit-Remaining-Month"))
                minute = number(response.headers.get("X-RateLimit-Remaining-Minute"))
                if remaining is not None and remaining <= 0:
                    reset = timestamp(response.headers.get("X-RateLimit-Reset-Month"))
                    self.quota_retry_after = (
                        max(3600, int((parse_timestamp(reset) - datetime.now(UTC)).total_seconds()))
                        if reset
                        else 86400
                    )
                elif minute is not None and minute <= 0:
                    self.quota_retry_after = 300
        except HTTPError as exc:
            kind = {401: "authentication", 403: "access denied", 429: "rate limit"}.get(
                exc.code, "provider failure"
            )
            raise OwlsError(
                f"Owls {endpoint}: {kind} (HTTP {exc.code})",
                retry_after=max(
                    3600 if exc.code in {401, 403} else 300,
                    retry_seconds(exc.headers.get("Retry-After") if exc.headers else None),
                ),
            ) from None
        except (URLError, TimeoutError, OSError):
            raise OwlsError(f"Owls {endpoint}: network failure") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OwlsError(f"Owls {endpoint}: invalid JSON") from None
        if not isinstance(payload, dict) or payload.get("success") is False:
            raise OwlsError(f"Owls {endpoint}: malformed response")
        return payload


def name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", normalized.casefold())


def team_aliases(sport: str, slate: list[dict[str, Any]]) -> dict[str, str]:
    names = {str(g[k]) for g in slate for k in ("home_team", "away_team") if g.get(k)}
    aliases = {name_key(n): n for n in names}
    if sport == "nfl":
        aliases.update({name_key(n): c for n, c in TEAM_NAME_TO_CODE.items()})
        aliases.update({name_key(c): c for c in TEAM_NAME_TO_CODE.values()})
        aliases.update({"lar": "LA", "wsh": "WAS", "jac": "JAX"})
    else:
        registry = read_json(PROJECT_ROOT / "data" / "market_team_aliases.json", {})
        aliases.update({name_key(n): c for n, c in registry.items()})
    return aliases


def match_event(
    event: dict[str, Any], sport: str, slate: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str]:
    aliases = team_aliases(sport, slate)
    home = aliases.get(name_key(str(event.get("home_team", ""))))
    away = aliases.get(name_key(str(event.get("away_team", ""))))
    if not home or not away:
        return None, "unknown_team"
    kickoff = timestamp(event.get("commence_time"))
    if not kickoff:
        return None, "invalid_kickoff"
    matches = []
    for game in slate:
        try:
            start = _game_kickoff(game)
        except (TypeError, ValueError):
            continue
        if (
            aliases.get(name_key(str(game.get("home_team", "")))) == home
            and aliases.get(name_key(str(game.get("away_team", "")))) == away
            and start is not None
            and start == parse_timestamp(kickoff)
        ):
            matches.append(game)
    if len(matches) != 1:
        return None, "ambiguous_game" if matches else "slate_or_kickoff_mismatch"
    return matches[0], "matched"


def parse_odds(
    payload: dict[str, Any],
    sport: str,
    slate: list[dict[str, Any]],
    captured_at: str,
    *,
    provider: str = "Owls Insight",
) -> dict[str, Any]:
    """One main line per book, median across books; never count alternate rungs."""
    data = payload.get("data")
    if not isinstance(data, dict) or payload.get("success") is False:
        raise OwlsError("Owls odds: malformed board")
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict) or not isinstance(meta.get("freshness", {}), dict):
        raise OwlsError("Owls odds: malformed metadata")
    games: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    seen = set()
    for bucket, events in data.items():
        if not isinstance(events, list):
            raise OwlsError("Owls odds: malformed sportsbook board")
        for event in events:
            if not isinstance(event, dict):
                diagnostics.append({"reason": "malformed_event", "book": bucket})
                continue
            matched, reason = match_event(event, sport, slate)
            if matched is None:
                diagnostics.append(
                    {
                        "reason": reason,
                        "book": bucket,
                        "event": event.get("eventId") or event.get("id"),
                        "home": event.get("home_team"),
                        "away": event.get("away_team"),
                        "kickoff": event.get("commence_time"),
                    }
                )
                continue
            game_id = str(matched["game_id"])
            # `id` alone is book-local. Older v1 boards omit eventId.
            event_id = event.get("eventId") or (
                f"{sport}:{event['away_team']}@{event['home_team']}-"
                f"{parse_timestamp(event['commence_time']):%Y%m%d}"
            )
            game = games.setdefault(
                game_id,
                {
                    "game_id": game_id,
                    "event_id": str(event_id),
                    "sport": sport,
                    "home_team": matched["home_team"],
                    "away_team": matched["away_team"],
                    "home_team_name": event["home_team"],
                    "away_team_name": event["away_team"],
                    "commence_time": timestamp(event["commence_time"]),
                    "books": {},
                    "captured_at": captured_at,
                },
            )
            if game["event_id"] != str(event_id):
                diagnostics.append(
                    {"reason": "conflicting_event_id", "game_id": game_id, "book": bucket}
                )
                continue
            books = event.get("bookmakers")
            if not isinstance(books, list):
                diagnostics.append({"reason": "missing_books", "game_id": game_id})
                continue
            for book in books:
                if not isinstance(book, dict) or not isinstance(book.get("markets"), list):
                    diagnostics.append({"reason": "malformed_book", "game_id": game_id})
                    continue
                key = str(book.get("key") or bucket)
                if (game_id, key) in seen:
                    diagnostics.append(
                        {"reason": "duplicate_book", "game_id": game_id, "book": key}
                    )
                    continue
                seen.add((game_id, key))
                row: dict[str, Any] = {
                    "sportsbook": key,
                    "source_timestamp": timestamp(book.get("last_update")),
                    "spread": None,
                    "spread_price": None,
                    "away_spread_price": None,
                    "moneyline": None,
                    "away_moneyline": None,
                    "total": None,
                    "over_price": None,
                    "under_price": None,
                }
                for market in book["markets"]:
                    if not isinstance(market, dict) or not isinstance(market.get("outcomes"), list):
                        diagnostics.append(
                            {"reason": "malformed_market", "game_id": game_id, "book": key}
                        )
                        continue
                    outcomes = {
                        o["name"]: o
                        for o in market["outcomes"]
                        if isinstance(o, dict) and isinstance(o.get("name"), str)
                    }
                    home = outcomes.get(event["home_team"], {})
                    away = outcomes.get(event["away_team"], {})
                    if market.get("key") == "spreads":
                        row.update(
                            spread=number(home.get("point")),
                            spread_price=number(home.get("price")),
                            away_spread_price=number(away.get("price")),
                        )
                        row["source_timestamp"] = (
                            timestamp(market.get("last_update")) or row["source_timestamp"]
                        )
                        away_point = number(away.get("point"))
                        if (
                            row["spread"] is not None
                            and away_point is not None
                            and abs(row["spread"] + away_point) > 0.01
                        ):
                            diagnostics.append(
                                {
                                    "reason": "inconsistent_spread_sides",
                                    "game_id": game_id,
                                    "book": key,
                                }
                            )
                            row["spread"] = None
                    elif market.get("key") == "totals":
                        row.update(
                            total=number(outcomes.get("Over", {}).get("point")),
                            over_price=number(outcomes.get("Over", {}).get("price")),
                            under_price=number(outcomes.get("Under", {}).get("price")),
                        )
                    elif market.get("key") == "h2h":
                        row.update(
                            moneyline=number(home.get("price")),
                            away_moneyline=number(away.get("price")),
                        )
                if all(row[k] is None for k in ("spread", "total", "moneyline")):
                    diagnostics.append(
                        {"reason": "missing_main_lines", "game_id": game_id, "book": key}
                    )
                    continue
                game["books"][key] = row
    for game in games.values():
        for market, line_key in (("spread", "home_spread"), ("total", "total")):
            rows = []
            for book in game["books"].values():
                if book[market] is None:
                    continue
                row = {line_key: book[market], "last_update": book["source_timestamp"]}
                price_keys = (
                    (("home_price", "spread_price"), ("away_price", "away_spread_price"))
                    if market == "spread"
                    else (("over_price", "over_price"), ("under_price", "under_price"))
                )
                for dest, source in price_keys:
                    if book[source] is not None:
                        row[dest] = book[source]
                rows.append(row)
            summary = _market_summary(rows, line_key)
            if summary:
                summary[line_key] = summary.pop("line")
                if market == "spread":
                    summary["market_home_margin"] = -summary[line_key]
            game[market] = summary
        spread_books = [b for b in game["books"].values() if b["spread"] is not None]
        times = [b["source_timestamp"] for b in spread_books]
        game["source_timestamp"] = min(times) if times and all(times) else None
        game["invalid_source_timestamp"] = any(t is None or t > captured_at for t in times)
        game["provider_stale"] = bool((meta.get("freshness") or {}).get("stale", False))
    return {
        "schema_version": 1,
        "provider": provider,
        "sport": sport,
        "snapshot_at": captured_at,
        "provider_timestamp": timestamp(meta.get("timestamp")),
        "games": list(games.values()),
        "diagnostics": diagnostics,
    }


def parse_splits(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(payload.get("data"), list) or payload.get("success") is False:
        raise OwlsError("Owls splits: malformed board")
    result: dict[str, dict[str, Any]] = {}
    for event in payload["data"]:
        if (
            not isinstance(event, dict)
            or not event.get("event_id")
            or not isinstance(event.get("splits"), list)
        ):
            raise OwlsError("Owls splits: malformed event")
        books = result.setdefault(str(event["event_id"]), {})
        for source in event["splits"]:
            if not isinstance(source, dict) or not source.get("book"):
                raise OwlsError("Owls splits: malformed book")
            key = "draftkings" if source["book"] == "dk" else str(source["book"])
            book: dict[str, Any] = {
                "sportsbook": key,
                "source_timestamp": timestamp(source.get("as_of")),
                "markets": {},
            }
            for market in ("spread", "moneyline", "total"):
                raw = source.get(market) or {}
                if not isinstance(raw, dict):
                    raise OwlsError("Owls splits: malformed market")
                sides = ("over", "under") if market == "total" else ("home", "away")
                values: dict[str, Any] = {}
                for side in sides:
                    tickets = number(raw.get(f"{side}_bets_pct"))
                    handle = number(raw.get(f"{side}_handle_pct"))
                    tickets = tickets if tickets is not None and 0 <= tickets <= 100 else None
                    handle = handle if handle is not None and 0 <= handle <= 100 else None
                    values[side] = {
                        "ticket_pct": tickets,
                        "handle_pct": handle,
                        "handle_minus_ticket": round(handle - tickets, 3)
                        if handle is not None and tickets is not None
                        else None,
                    }
                for field in ("ticket_pct", "handle_pct"):
                    pair = [values[s][field] for s in sides]
                    if all(v is not None for v in pair) and abs(sum(pair) - 100) > 1:
                        for side in sides:
                            values[side][field] = None
                            values[side]["handle_minus_ticket"] = None
                ticket_side = next((s for s in sides if (values[s]["ticket_pct"] or 0) > 50), None)
                money_side = next((s for s in sides if (values[s]["handle_pct"] or 0) > 50), None)
                values.update(
                    majority_ticket_side=ticket_side,
                    majority_money_side=money_side,
                    ticket_money_disagreement=ticket_side != money_side
                    if ticket_side and money_side
                    else None,
                )
                book["markets"][market] = values
            books[key] = book
    return result
