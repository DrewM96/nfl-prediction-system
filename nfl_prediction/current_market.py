"""Shared backend cache and append-only market observations; never runs a model."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg

from . import market_postgres
from .config import MARKET_PRIVATE_DIR
from .odds import _game_kickoff, attach_market_consensus, parse_timestamp
from .owls import OwlsClient, OwlsError, parse_odds, parse_splits, timestamp

POLL_SECONDS = 300
STALE_SECONDS = 900
SPLIT_STALE_SECONDS = 3600


def database_path() -> Path:
    return Path(os.environ.get("GRIDLINE_MARKET_DB", str(MARKET_PRIVATE_DIR / "market.sqlite3")))


def market_provider() -> str:
    provider = os.environ.get("GRIDLINE_MARKET_PROVIDER", "owls").lower()
    if provider not in {"owls", "legacy"}:
        raise ValueError("GRIDLINE_MARKET_PROVIDER must be owls or legacy")
    return provider


def age_seconds(value: str | None, now: datetime) -> float | None:
    stamp = timestamp(value)
    age = (now - parse_timestamp(stamp)).total_seconds() if stamp else None
    return age if age is not None and age >= 0 else None


def stale_at(value: str | None, now: datetime, *, max_age: int = STALE_SECONDS) -> bool:
    age = age_seconds(value, now)
    return age is None or age > max_age


class MarketStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else database_path()
        # An explicit CLI/test path selects SQLite. Otherwise all dynos use PG.
        self.database_url = (
            os.environ.get("GRIDLINE_MARKET_DATABASE_URL") or os.environ.get("DATABASE_URL")
            if path is None
            else None
        )

    @property
    def configuration_error(self) -> str | None:
        if os.environ.get("DYNO") and not self.database_url:
            return "Persistent market database is not configured for Heroku"
        return None

    @contextmanager
    def writer(self):
        if self.database_url:
            with market_postgres.writer(self.database_url) as db:
                yield db
        else:
            with closing(self.connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                yield db

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=1)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS market_cache (
                sport TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS poll_state (
                key TEXT PRIMARY KEY, next_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS market_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL, game_id TEXT NOT NULL, sport TEXT NOT NULL,
                sportsbook TEXT NOT NULL, captured_at TEXT NOT NULL,
                source_timestamp TEXT, spread REAL, spread_price REAL,
                moneyline REAL, total REAL, splits_json TEXT,
                payload TEXT NOT NULL, material_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS market_observation_latest
                ON market_observations(sport, event_id, sportsbook, id DESC);
            CREATE TABLE IF NOT EXISTS prop_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL, sportsbook TEXT NOT NULL, player_key TEXT NOT NULL,
                category TEXT NOT NULL, game_id TEXT NOT NULL, captured_at TEXT NOT NULL,
                payload TEXT NOT NULL, material_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS prop_observation_latest
                ON prop_observations(event_id, sportsbook, player_key, category, id DESC);
            CREATE TRIGGER IF NOT EXISTS prop_observations_no_update
                BEFORE UPDATE ON prop_observations
                BEGIN SELECT RAISE(ABORT, 'Prop history is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS prop_observations_no_delete
                BEFORE DELETE ON prop_observations
                BEGIN SELECT RAISE(ABORT, 'Prop history is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS market_observations_no_update
                BEFORE UPDATE ON market_observations
                BEGIN SELECT RAISE(ABORT, 'Market history is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS market_observations_no_delete
                BEFORE DELETE ON market_observations
                BEGIN SELECT RAISE(ABORT, 'Market history is append-only'); END;
            PRAGMA user_version=2;
        """)
        return connection

    def read(self, sport: str) -> dict[str, Any]:
        if self.configuration_error:
            return {"games": [], "odds_error": self.configuration_error}
        if not self.database_url and not self.path.exists():
            return {"games": [], "odds_error": "Market worker has not populated the cache"}
        try:
            if self.database_url:
                payload = market_postgres.read(self.database_url, sport)
            else:
                with closing(
                    sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=1)
                ) as db:
                    row = db.execute(
                        "SELECT payload FROM market_cache WHERE sport=?", (sport,)
                    ).fetchone()
                payload = json.loads(row[0]) if row else None
            return payload or {"games": [], "odds_error": "No cached market for sport"}
        except (sqlite3.Error, psycopg.Error, OSError, ValueError):
            return {"games": [], "odds_error": "Market cache unavailable"}

    @staticmethod
    def observe(db: Any, game: dict[str, Any], captured_at: str) -> None:
        for key in set(game.get("books", {})) | set(game.get("splits", {})):
            odds = game.get("books", {}).get(key, {})
            splits = game.get("splits", {}).get(key)
            material = {
                "odds": {k: v for k, v in odds.items() if k != "source_timestamp"},
                "splits": splits.get("markets") if splits else None,
                "game_id": game["game_id"],
            }
            digest = hashlib.sha256(
                json.dumps(material, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            previous = db.execute(
                "SELECT material_hash FROM market_observations WHERE sport=? AND event_id=? AND sportsbook=? ORDER BY id DESC LIMIT 1",
                (game["sport"], game["event_id"], key),
            ).fetchone()
            if previous and previous[0] == digest:
                continue
            db.execute(
                """INSERT INTO market_observations
                (event_id,game_id,sport,sportsbook,captured_at,source_timestamp,
                 spread,spread_price,moneyline,total,splits_json,payload,material_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    game["event_id"],
                    game["game_id"],
                    game["sport"],
                    key,
                    captured_at,
                    odds.get("source_timestamp"),
                    odds.get("spread"),
                    odds.get("spread_price"),
                    odds.get("moneyline"),
                    odds.get("total"),
                    json.dumps(splits),
                    json.dumps({"odds": odds, "splits": splits}, allow_nan=False),
                    digest,
                ),
            )


def poll_market(
    sport: str,
    slate: list[dict[str, Any]],
    *,
    store: MarketStore | None = None,
    client: OwlsClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Transactional single writer, persisted cadence/backoff shared by all workers."""
    store = store or MarketStore()
    if store.configuration_error:
        return {"games": [], "odds_error": store.configuration_error}
    client = client or OwlsClient()
    now = now or datetime.now(UTC)
    captured = now.isoformat()
    try:
        with store.writer() as db:
            old_row = db.execute(
                "SELECT payload FROM market_cache WHERE sport=?", (sport,)
            ).fetchone()
            board = json.loads(old_row[0]) if old_row else {"games": []}
            gates = db.execute(
                "SELECT key,next_at FROM poll_state WHERE key IN (?, 'account')", (sport,)
            ).fetchall()
            for key, next_at in gates:
                if parse_timestamp(next_at) > now:
                    if key == "account":
                        return {
                            **board,
                            "odds_error": board.get("odds_error")
                            or "Provider account backoff active",
                            "splits_error": board.get("splits_error")
                            or "Provider account backoff active",
                        }
                    return board
            db.execute(
                "INSERT INTO poll_state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET next_at=excluded.next_at",
                (sport, (now + timedelta(seconds=POLL_SECONDS)).isoformat()),
            )
            if not slate:
                board["odds_error"] = "No GRIDLINE slate available"
            else:
                previous = {g["game_id"]: g for g in board.get("games", [])}
                try:
                    fresh = parse_odds(client.get(sport, "odds"), sport, slate, captured)
                    fresh_ids = {g["game_id"] for g in fresh["games"] if g.get("spread")}
                    merged = []
                    for game in fresh["games"]:
                        old = previous.get(game["game_id"], {})
                        if old.get("event_id") != game["event_id"]:
                            old = {}
                        if not game.get("spread") and old.get("spread"):
                            game = {**old, "unavailable": "Spread missing from latest board"}
                        else:
                            game["splits"] = copy.deepcopy(old.get("splits", {}))
                            game["missing_books"] = sorted(
                                set(old.get("books", {})) - set(game.get("books", {}))
                            )
                        merged.append(game)
                    merged_ids = {g["game_id"] for g in merged}
                    merged.extend(
                        {**g, "unavailable": "Game missing from latest board"}
                        for gid, g in previous.items()
                        if gid not in merged_ids
                    )
                    board = {**fresh, "games": merged, "odds_error": None}
                    board["missing_games"] = sorted(
                        str(g["game_id"]) for g in slate if str(g["game_id"]) not in fresh_ids
                    )
                except OwlsError as exc:
                    board["odds_error"] = str(exc)
                    board["splits_error"] = "Splits not refreshed because odds request failed"
                    db.execute(
                        "INSERT INTO poll_state VALUES ('account',?) ON CONFLICT(key) DO UPDATE SET next_at=excluded.next_at",
                        ((now + timedelta(seconds=exc.retry_after)).isoformat(),),
                    )
                # Account-wide authentication/rate-limit backoff also stops the second request.
                quota_wait = getattr(client, "quota_retry_after", 0)
                if quota_wait:
                    board["splits_error"] = "Provider quota exhausted; cached splits retained"
                if not board.get("odds_error") and not quota_wait:
                    try:
                        splits = parse_splits(client.get(sport, "splits"))
                        board["splits_error"] = None
                        known_ids = {g["event_id"] for g in board["games"]}
                        board["unmatched_split_events"] = sorted(set(splits) - known_ids)
                        for game in board["games"]:
                            incoming = splits.get(game["event_id"], {})
                            retained = {
                                key: {**value, "unavailable": True}
                                for key, value in game.get("splits", {}).items()
                                if key not in incoming
                            }
                            game["splits"] = {**retained, **incoming}
                    except OwlsError as exc:
                        board["splits_error"] = str(exc)
                        db.execute(
                            "INSERT INTO poll_state VALUES ('account',?) ON CONFLICT(key) DO UPDATE SET next_at=excluded.next_at",
                            ((now + timedelta(seconds=exc.retry_after)).isoformat(),),
                        )
                quota_wait = getattr(client, "quota_retry_after", 0)
                if quota_wait:
                    db.execute(
                        "INSERT INTO poll_state VALUES ('account',?) ON CONFLICT(key) DO UPDATE SET next_at=excluded.next_at",
                        ((now + timedelta(seconds=quota_wait)).isoformat(),),
                    )
                for game in board.get("games", []):
                    # Do not timestamp old odds as a new observation on failed/missing polls.
                    if not board.get("odds_error") and not game.get("unavailable"):
                        store.observe(db, game, captured)
            board["last_attempt_at"] = captured
            db.execute(
                "INSERT INTO market_cache VALUES (?,?) ON CONFLICT(sport) DO UPDATE SET payload=excluded.payload",
                (sport, json.dumps(board, allow_nan=False)),
            )
            return board
    except (sqlite3.Error, psycopg.Error, OSError, ValueError):
        return {**store.read(sport), "odds_error": "Market cache busy or unavailable"}


def current_context(
    forecast: dict[str, Any], board: dict[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    """Read-only projection of current context; frozen dictionaries are never mutated."""
    now = now or datetime.now(UTC)
    checked = {
        "last_attempt_at": board.get("last_attempt_at"),
        "checked_age_seconds": age_seconds(board.get("last_attempt_at"), now),
    }
    market = next(
        (g for g in board.get("games", []) if str(g["game_id"]) == str(forecast.get("game_id"))),
        None,
    )
    if market is None:
        return {
            **checked,
            "status": "unavailable",
            "reason": board.get("odds_error") or "Game not matched",
            "splits": {},
        }
    try:
        kickoff = _game_kickoff(forecast)
    except (ValueError, TypeError):
        kickoff = None
    if (
        kickoff is None
        or timestamp(market.get("commence_time")) != kickoff.isoformat()
        or any(market.get(k) != forecast.get(k) for k in ("home_team", "away_team"))
    ):
        return {
            **checked,
            "status": "unavailable",
            "reason": "Game identity or kickoff mismatch",
            "splits": {},
        }
    current = (market.get("spread") or {}).get("home_spread")
    frozen = (forecast.get("market_consensus") or {}).get("spread") or {}
    original = frozen.get("home_spread")
    if original is None and frozen.get("market_home_margin") is not None:
        original = -float(frozen["market_home_margin"])
    reasons = []
    if board.get("odds_error"):
        reasons.append(board["odds_error"])
    if market.get("unavailable"):
        reasons.append(market["unavailable"])
    if market.get("provider_stale"):
        reasons.append("Provider reports stale odds")
    if market.get("invalid_source_timestamp"):
        reasons.append("A contributing book has a missing or future timestamp")
    if stale_at(market.get("captured_at"), now) or stale_at(market.get("source_timestamp"), now):
        reasons.append("Odds timestamp is old, missing, or in the future")
    if kickoff <= now:
        reasons.append("Kickoff reached; pregame projection comparison only")
    if current is None:
        reasons.append("No spread available")
    splits = copy.deepcopy(market.get("splits", {}))
    for book in splits.values():
        book["source_age_seconds"] = age_seconds(book.get("source_timestamp"), now)
        book["stale"] = bool(
            board.get("splits_error")
            or book.get("unavailable")
            or stale_at(book.get("source_timestamp"), now, max_age=SPLIT_STALE_SECONDS)
        )
    agreement: dict[str, dict[str, str]] = {}
    for kind in ("spread", "moneyline", "total"):
        agreement[kind] = {}
        for field in ("majority_ticket_side", "majority_money_side"):
            sides = [
                b["markets"][kind][field]
                for b in splits.values()
                if not b["stale"] and b["markets"][kind][field]
            ]
            agreement[kind][field] = (
                ("agree" if len(set(sides)) == 1 else "disagree")
                if len(sides) >= 2
                else "insufficient_data"
            )
    margin = forecast.get("predicted_home_margin")
    return {
        **checked,
        "status": "stale" if reasons else "fresh",
        "reason": "; ".join(reasons),
        "provider": board.get("provider", "Owls Insight"),
        "event_id": market["event_id"],
        "home_spread": current,
        "source_timestamp": market.get("source_timestamp"),
        "captured_at": market.get("captured_at"),
        "movement": round(current - original, 3)
        if current is not None and original is not None
        else None,
        "current_home_edge": round(float(margin) + current, 3)
        if margin is not None and current is not None
        else None,
        "splits": splits,
        "splits_error": board.get("splits_error"),
        "sportsbook_agreement": agreement,
        "missing_books": market.get("missing_books", []),
    }


def freeze_market_context(
    predictions: list[dict[str, Any]],
    sport: str,
    *,
    as_of: datetime,
    store: MarketStore | None = None,
) -> list[dict[str, Any]]:
    """Call once AFTER projections, before recording a new immutable batch."""
    if market_provider() == "legacy":
        return predictions
    board = (store or MarketStore()).read(sport)
    eligible: list[dict[str, Any]] = []
    for prediction in predictions:
        context = current_context(prediction, board, now=as_of)
        if context["status"] == "fresh":
            eligible.extend(g for g in board["games"] if g["game_id"] == str(prediction["game_id"]))
    snapshot = {**board, "games": eligible}
    # Existing attachment preserves all predictive fields and benchmark conventions.
    enriched = attach_market_consensus(
        predictions, snapshot, as_of=as_of, max_age=timedelta(seconds=STALE_SECONDS)
    )
    by_id = {g["game_id"]: g for g in eligible}
    for prediction in enriched:
        prediction["forecast_at"] = as_of.isoformat()
        prediction["market_context_status"] = (
            "captured" if prediction.get("market_consensus") else "unavailable"
        )
        market = by_id.get(str(prediction.get("game_id")))
        if market and prediction.get("market_consensus"):
            prediction["market_consensus"]["event_id"] = market["event_id"]
            prediction["market_consensus"]["source_timestamp"] = market["source_timestamp"]
    return enriched
