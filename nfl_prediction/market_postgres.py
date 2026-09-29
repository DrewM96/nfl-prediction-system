"""Persistent market storage for hosts whose web/worker filesystems are isolated."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

# One transaction lock coordinates account-wide quota gates across all dynos.
MARKET_LOCK = 714620190813

SCHEMA = """
CREATE SCHEMA IF NOT EXISTS gridline_market;
SET LOCAL search_path TO gridline_market;
CREATE TABLE IF NOT EXISTS market_cache (sport TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS poll_state (key TEXT PRIMARY KEY, next_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS market_openings (
    sport TEXT NOT NULL, game_id TEXT NOT NULL,
    open_home_spread DOUBLE PRECISION NOT NULL, open_snapshot_at TEXT NOT NULL,
    PRIMARY KEY (sport, game_id)
);
CREATE TABLE IF NOT EXISTS market_observations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id TEXT NOT NULL, game_id TEXT NOT NULL, sport TEXT NOT NULL,
    sportsbook TEXT NOT NULL, captured_at TEXT NOT NULL,
    source_timestamp TEXT, spread DOUBLE PRECISION, spread_price DOUBLE PRECISION,
    moneyline DOUBLE PRECISION, total DOUBLE PRECISION, splits_json TEXT,
    payload TEXT NOT NULL, material_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS market_observation_latest
    ON market_observations(sport, event_id, sportsbook, id DESC);
CREATE TABLE IF NOT EXISTS prop_observations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id TEXT NOT NULL, sportsbook TEXT NOT NULL, player_key TEXT NOT NULL,
    category TEXT NOT NULL, game_id TEXT NOT NULL, captured_at TEXT NOT NULL,
    payload TEXT NOT NULL, material_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS prop_observation_latest
    ON prop_observations(event_id, sportsbook, player_key, category, id DESC);
CREATE OR REPLACE FUNCTION reject_market_history_change() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'Market history is append-only'; END;
$$;
CREATE OR REPLACE TRIGGER market_observations_immutable
    BEFORE UPDATE OR DELETE OR TRUNCATE ON market_observations
    FOR EACH STATEMENT EXECUTE FUNCTION reject_market_history_change();
CREATE OR REPLACE TRIGGER prop_observations_immutable
    BEFORE UPDATE OR DELETE OR TRUNCATE ON prop_observations
    FOR EACH STATEMENT EXECUTE FUNCTION reject_market_history_change();
"""


def connect(url: str) -> psycopg.Connection:
    options: dict[str, Any] = conninfo_to_dict(url)
    options.setdefault("sslmode", "require")
    options["connect_timeout"] = 5
    connection = psycopg.connect(make_conninfo(**options))
    try:
        connection.execute("SET LOCAL lock_timeout = '1000ms'")
        connection.execute("SET LOCAL statement_timeout = '10000ms'")
        connection.execute("SET LOCAL search_path TO gridline_market")
    except BaseException:
        connection.close()
        raise
    return connection


class Session:
    """Keep the store's fixed, parameterized SQL portable between SQLite and PG."""

    def __init__(self, connection: psycopg.Connection):
        self.connection = connection

    def execute(self, query: str, parameters: tuple = ()) -> Any:
        return self.connection.execute(query.replace("?", "%s"), parameters)


@contextmanager
def writer(url: str):
    with connect(url) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (MARKET_LOCK,))
        connection.execute(SCHEMA)
        yield Session(connection)


def read(url: str, sport: str) -> dict[str, Any] | None:
    import json

    with connect(url) as connection:
        row = connection.execute(
            "SELECT payload FROM market_cache WHERE sport=%s", (sport,)
        ).fetchone()
    return json.loads(row[0]) if row else None
