"""Prospective ticket/handle disagreement signals against published forecasts."""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .odds import _game_kickoff, parse_timestamp
from .owls import number, timestamp
from .results_tracker import grade_pick, pick_side

POLICY_VERSION = "handle-model-v1"
MIN_SPLIT_BOOKS = 2
MIN_HANDLE_PCT = 65


def published_forecasts(payload: dict | list) -> list[dict]:
    if isinstance(payload, list):
        return payload
    return [
        {
            **row,
            "forecast_created_at": payload.get("created_at"),
            "forecast_run_id": payload.get("run_id"),
            "forecast_model_hash": payload.get("model_hash"),
        }
        for row in payload.get("predictions", payload.get("games", []))
    ]


def _recent(value: Any, now: datetime, seconds: int) -> bool:
    stamp = timestamp(value)
    return stamp is not None and 0 <= (now - parse_timestamp(stamp)).total_seconds() <= seconds


def qualifying_signals(
    forecast: dict[str, Any], board: dict[str, Any], *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Require fresh odds and paired splits; never reconstruct past signals."""
    now = now or datetime.now(UTC)
    if board.get("odds_error") or board.get("splits_error"):
        return []
    game = next(
        (g for g in board.get("games", []) if str(g["game_id"]) == str(forecast.get("game_id"))),
        None,
    )
    if not game or game.get("unavailable") or game.get("provider_stale"):
        return []
    try:
        kickoff = _game_kickoff(forecast)
        published = parse_timestamp(forecast.get("forecast_created_at") or forecast["forecast_at"])
    except (KeyError, TypeError, ValueError):
        return []
    if (
        kickoff is None
        or not published <= now < kickoff
        or timestamp(game.get("commence_time")) != kickoff.isoformat()
        or any(game.get(key) != forecast.get(key) for key in ("home_team", "away_team"))
        or not _recent(game.get("captured_at"), now, 900)
    ):
        return []
    captured = parse_timestamp(game["captured_at"])
    records = []
    for kind in ("spread", "total"):
        sides = ("home", "away") if kind == "spread" else ("over", "under")
        splits = {}
        for key, book in game.get("splits", {}).items():
            values = book.get("markets", {}).get(kind, {})
            if book.get("unavailable") or not _recent(book.get("source_timestamp"), now, 3600):
                continue
            if parse_timestamp(book["source_timestamp"]) > captured:
                continue
            pairs = {
                field: [number(values.get(side, {}).get(field)) for side in sides]
                for field in ("ticket_pct", "handle_pct")
            }
            if any(
                any(v is None or not 0 <= v <= 100 for v in pair)
                or abs(sum(v for v in pair if v is not None) - 100) > 1
                for pair in pairs.values()
            ):
                continue
            splits[key] = {
                "source_timestamp": book["source_timestamp"],
                "sides": copy.deepcopy(values),
            }
        if len(splits) < MIN_SPLIT_BOOKS:
            continue
        averages = {
            side: {
                field: statistics.mean(b["sides"][side][field] for b in splits.values())
                for field in ("ticket_pct", "handle_pct")
            }
            for side in sides
        }
        public_sides = [s for s in sides if averages[s]["ticket_pct"] > 50]
        if len(public_sides) != 1:
            continue
        public_side = public_sides[0]
        handle_side = sides[1] if public_side == sides[0] else sides[0]
        if averages[handle_side]["handle_pct"] < MIN_HANDLE_PCT:
            continue
        odds = {}
        for key, book in game.get("books", {}).items():
            line = number(book.get(kind))
            # Older caches lack market-specific total timestamps; wait for the next poll.
            source = book.get(f"{kind}_source_timestamp")
            if kind == "spread" and source is None:
                source = book.get("source_timestamp")
            if line is None or not _recent(source, now, 900) or parse_timestamp(source) > captured:
                continue
            odds[key] = {"line": line, "source_timestamp": source}
        if not odds:
            continue
        line = statistics.median(b["line"] for b in odds.values())
        projection = number(
            forecast.get("predicted_home_margin")
            if kind == "spread"
            else forecast.get("predicted_total", forecast.get("total"))
        )
        reference = -line if kind == "spread" else line
        model_side = pick_side(projection, reference)
        if model_side != (1 if handle_side == sides[0] else -1):
            continue
        record = {
            "policy": POLICY_VERSION,
            "sport": game["sport"],
            "game_id": str(forecast["game_id"]),
            "event_id": game["event_id"],
            "season": forecast.get("season", kickoff.year),
            "week": forecast.get("week"),
            "home_team": forecast["home_team"],
            "away_team": forecast["away_team"],
            "kickoff": kickoff.isoformat(),
            "observed_at": now.isoformat(),
            "market_captured_at": game["captured_at"],
            "forecast_at": published.isoformat(),
            "forecast_run_id": forecast.get("forecast_run_id"),
            "model_hash": forecast.get("forecast_model_hash"),
            "market": kind,
            "public_side": public_side,
            "handle_side": handle_side,
            "ticket_pct": averages[public_side]["ticket_pct"],
            "handle_pct": averages[handle_side]["handle_pct"],
            "averages": averages,
            "line": line,
            "projection": projection,
            "model_edge": abs(projection - reference),
            "split_books": splits,
            "odds_books": odds,
        }
        record["signal_id"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        records.append(record)
    return records


def signal_label(signal: dict[str, Any]) -> str:
    side, line = signal["handle_side"], signal["line"]
    if signal["market"] == "total":
        return f"{side.title()} {line:g}"
    team = signal[f"{side}_team"]
    return f"{team} {line if side == 'home' else -line:+g}"


def first_signal_records(observations: list[dict], results: pd.DataFrame) -> pd.DataFrame:
    """Grade each game/market once, using its first recorded signal and line."""
    if not observations:
        return pd.DataFrame()
    rows = (
        pd.DataFrame(observations)
        .sort_values(["observed_at", "signal_id"])
        .drop_duplicates(["sport", "game_id", "market"], keep="first")
        .copy()
    )
    outcomes = {}
    if not results.empty:
        settled = results[results.scored_at.notna()].sort_values("scored_at")
        outcomes = {str(r["game_id"]): r for r in settled.to_dict("records")}
    grades = []
    for row in rows.to_dict("records"):
        actual = outcomes.get(row["game_id"], {})
        grades.append(
            grade_pick(
                row["projection"],
                -row["line"] if row["market"] == "spread" else row["line"],
                actual.get("actual_margin" if row["market"] == "spread" else "actual_total"),
                actual.get("status", "scheduled"),
            )
        )
    rows["result"] = grades
    return rows.reset_index(drop=True)
