"""Backend-only props/depth refreshes sharing persistent cache and account gates."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
import psycopg

from .current_market import MarketStore
from .odds import _game_kickoff, parse_timestamp
from .owls import OwlsClient, OwlsError
from .player_props import parse_props, prop_key


def parse_depth(frame: pd.DataFrame, now: datetime) -> dict:
    required = {
        "dt",
        "team",
        "player_name",
        "gsis_id",
        "pos_abb",
        "pos_grp",
        "pos_slot",
        "pos_rank",
    }
    if not required.issubset(frame.columns):
        raise OwlsError("Depth chart schema is unavailable")
    frame = frame.copy()
    frame["source_time"] = pd.to_datetime(frame["dt"], utc=True, errors="coerce")
    frame = frame[frame["source_time"].le(now)]
    frame = frame[frame["source_time"].eq(frame.groupby("team")["source_time"].transform("max"))]
    frame = frame[frame["pos_abb"].isin(["QB", "RB", "FB", "WR", "TE"])]
    frame["rank"] = pd.to_numeric(frame["pos_rank"], errors="coerce")
    frame = frame[frame["rank"].ge(1) & frame["pos_slot"].notna()]
    # WR rank is across the position group: rank 2/3 can each lead a distinct slot.
    slots = ["team", "pos_grp", "pos_slot"]
    first = frame[frame["rank"].eq(frame.groupby(slots)["rank"].transform("min"))]
    first = first[
        first.groupby(slots)["gsis_id"].transform(lambda ids: ids.nunique(dropna=False)).eq(1)
    ].dropna(subset=["gsis_id"])
    ids = set(zip(first["team"], first["gsis_id"], strict=False))
    players = []
    for row in (
        frame.dropna(subset=["gsis_id", "player_name"])
        .drop_duplicates(["team", "gsis_id"])
        .to_dict("records")
    ):
        players.append(
            {
                "team": str(row["team"]),
                "player_id": str(row["gsis_id"]),
                "player_name": str(row["player_name"]),
                "position": str(row["pos_abb"]),
                "starter": (row["team"], row["gsis_id"]) in ids,
                "source_timestamp": row["source_time"].isoformat(),
            }
        )
    if not players:
        raise OwlsError("No current depth chart players returned")
    return {"players": players, "source": "nflverse / ESPN depth charts"}


def fetch_depth(season: int, now: datetime) -> dict:
    url = f"https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{int(season)}.parquet"
    try:
        with urlopen(Request(url, headers={"User-Agent": "GRIDLINE/4.0"}), timeout=30) as response:
            data = response.read()
        frame = pd.read_parquet(
            io.BytesIO(data),
            columns=[
                "dt",
                "team",
                "player_name",
                "gsis_id",
                "pos_abb",
                "pos_grp",
                "pos_slot",
                "pos_rank",
            ],
            dtype_backend="pyarrow",
        )
        return parse_depth(frame, now)
    except Exception:
        raise OwlsError("Depth chart refresh unavailable", retry_after=3600) from None


def observe_props(db: Any, rows: list[dict], captured_at: str) -> None:
    for row in rows:
        if row.get("unavailable"):
            continue
        material = {k: row.get(k) for k in ("line", "over_price", "under_price", "team", "game_id")}
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
        key = prop_key(row)
        previous = db.execute(
            "SELECT material_hash FROM prop_observations WHERE event_id=? AND sportsbook=? AND player_key=? AND category=? ORDER BY id DESC LIMIT 1",
            key,
        ).fetchone()
        if previous and previous[0] == digest:
            continue
        db.execute(
            "INSERT INTO prop_observations (event_id,sportsbook,player_key,category,game_id,captured_at,payload,material_hash) VALUES (?,?,?,?,?,?,?,?)",
            (*key, row["game_id"], captured_at, json.dumps(row, allow_nan=False), digest),
        )


def refresh_context(
    key: str,
    loader: Callable[[], dict],
    *,
    store: MarketStore,
    now: datetime,
    interval: int,
    client: OwlsClient | None = None,
) -> dict:
    if store.configuration_error:
        return {"error": store.configuration_error}
    try:
        with store.writer() as db:
            found = db.execute("SELECT payload FROM market_cache WHERE sport=?", (key,)).fetchone()
            board = json.loads(found[0]) if found else {}
            gates = db.execute(
                "SELECT key,next_at FROM poll_state WHERE key IN (?,?) ORDER BY key",
                (key, "account" if client else key),
            ).fetchall()
            for gate, next_at in gates:
                if parse_timestamp(next_at) > now:
                    if gate == "account":
                        board["error"] = board.get("error") or "Provider account backoff active"
                        db.execute(
                            "INSERT INTO market_cache VALUES (?,?) ON CONFLICT(sport) DO UPDATE SET payload=excluded.payload",
                            (key, json.dumps(board, allow_nan=False)),
                        )
                    return board
            next_at = now + timedelta(seconds=interval)
            try:
                fresh = loader()
                if key == "nfl_props":
                    current = {prop_key(row): row for row in fresh["rows"]}
                    old = {
                        prop_key(row): {**row, "unavailable": True}
                        for row in board.get("rows", [])
                        if parse_timestamp(row["commence_time"]) > now - timedelta(days=2)
                    }
                    fresh["rows"] = list((old | current).values())
                    observe_props(db, fresh["rows"], now.isoformat())
                board = {**fresh, "error": None}
            except OwlsError as exc:
                board["error"] = str(exc)
                next_at = now + timedelta(seconds=max(interval, exc.retry_after))
                if client:
                    db.execute(
                        "INSERT INTO poll_state VALUES ('account',?) ON CONFLICT(key) DO UPDATE SET next_at=excluded.next_at",
                        (next_at.isoformat(),),
                    )
            wait = getattr(client, "quota_retry_after", 0)
            if wait:
                db.execute(
                    "INSERT INTO poll_state VALUES ('account',?) ON CONFLICT(key) DO UPDATE SET next_at=excluded.next_at",
                    ((now + timedelta(seconds=wait)).isoformat(),),
                )
            board["last_attempt_at"] = now.isoformat()
            db.execute(
                "INSERT INTO poll_state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET next_at=excluded.next_at",
                (key, next_at.isoformat()),
            )
            db.execute(
                "INSERT INTO market_cache VALUES (?,?) ON CONFLICT(sport) DO UPDATE SET payload=excluded.payload",
                (key, json.dumps(board, allow_nan=False)),
            )
            return board
    except (sqlite3.Error, psycopg.Error, OSError, ValueError):
        return {**store.read(key), "error": "Props cache busy or unavailable"}


def poll_props(
    slate: list[dict],
    *,
    store: MarketStore,
    client: OwlsClient | None = None,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    client = client or OwlsClient()

    def load() -> dict:
        try:
            upcoming = [game for game in slate if (start := _game_kickoff(game)) and start > now]
        except (TypeError, ValueError):
            raise OwlsError("Props slate has an invalid kickoff timestamp") from None
        if not upcoming:
            return {"rows": [], "diagnostics": {"no_upcoming_forecasts": 1}}
        return parse_props(client.get("nfl", "props"), upcoming, now.isoformat())

    return refresh_context("nfl_props", load, store=store, now=now, interval=900, client=client)


def poll_depth(season: int, *, store: MarketStore, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    return refresh_context(
        "nfl_depth", lambda: fetch_depth(season, now), store=store, now=now, interval=21600
    )
