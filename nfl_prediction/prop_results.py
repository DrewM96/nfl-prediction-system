"""Prospective player forecast archive and append-only official-stat settlements."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from .io import atomic_write_json, read_json
from .odds import parse_timestamp
from .player_props import PROP_MODELS, comparisons, projection_rows
from .results import prediction_batches
from .results_tracker import grade_pick, numeric


def refresh_player_results(
    root: str | Path, schedules: pd.DataFrame, season: int, *, now: datetime
) -> int:
    if not any(b.get("player_predictions") for b in prediction_batches(root)):
        return 0
    import nflreadpy as nfl

    try:
        nfl.clear_cache("stats_player")
        stats = nfl.load_player_stats(season).to_pandas()
    except (ConnectionError, OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Player results awaiting nflverse stats: %s", exc)
        return 0
    return settle_player_predictions(root, stats, schedules, now=now)


def freeze_player_predictions(
    snapshot: dict, models: dict, manifest: dict, board: dict, depth: dict, *, now: datetime
) -> list[dict]:
    """Freeze every published projection; attach only fresh, pregame consensus lines."""
    projections = projection_rows(snapshot, models, manifest)
    matches = {}
    for category in PROP_MODELS:
        for row in comparisons(projections, board, depth, category, now=now):
            matches[(row["game_id"], row["player_id"], category)] = row
    games = {str(g["game_id"]): g for g in snapshot.get("schedule", [])}
    result = []
    for projection in projections:
        if parse_timestamp(projection["commence_time"]) <= now:
            continue
        game = games[projection["game_id"]]
        match = matches.get(
            (projection["game_id"], projection["player_id"], projection["category"]), {}
        )
        result.append(
            {
                **projection,
                "player_name": match.get("player_name", projection["player_name"]),
                "season": int(manifest["prediction_season"]),
                "week": int(game["week"]),
                "market_line": match.get("line"),
                "market_at": match.get("source_timestamp"),
                "market_books": match.get("books", 0),
                "market_captured_at": now.isoformat() if match else None,
            }
        )
    return result


def prop_id(row: dict) -> str:
    return f"{row['game_id']}:{row['player_id']}:{row['category']}"


def prop_events(root: str | Path, run_id: str) -> list[dict]:
    if Path(run_id).name != run_id:
        raise ValueError("Invalid prediction run ID")
    return [read_json(p) for p in sorted((Path(root) / "prop_settlements" / run_id).glob("*.json"))]


def latest_prop_results(root: str | Path, run_id: str) -> dict:
    latest = {}
    for event in prop_events(root, run_id):
        for row in event["results"]:
            latest[row["prop_id"]] = {
                **row,
                "revision": event["revision"],
                "scored_at": event["scored_at"],
            }
    return latest


def settle_player_predictions(
    root: str | Path, stats: pd.DataFrame, schedules: pd.DataFrame, *, now: datetime
) -> int:
    """Use published weekly stats by stable player ID; missing players stay unresolved.

    Eight hours after kickoff is the same conservative finality guard as the daily
    game settler. No historical predictions or market lines are reconstructed.
    """
    if stats.empty or schedules.empty:
        return 0
    from .odds import _game_kickoff

    games = {}
    for game in schedules.to_dict("records"):
        if game.get("status") in {"cancelled", "canceled", "postponed", "in_progress"}:
            continue
        kickoff = _game_kickoff(game)
        if (
            kickoff
            and now - kickoff >= timedelta(hours=8)
            and numeric(game.get("home_score")) is not None
            and numeric(game.get("away_score")) is not None
        ):
            games[str(game["game_id"])] = game
    lookup: dict[tuple, list] = {}
    for row in stats.to_dict("records"):
        if row.get("season_type", "REG") != "REG":
            continue
        team = row.get("team", row.get("recent_team"))
        key = (row.get("season"), row.get("week"), str(row.get("player_id")), team)
        lookup.setdefault(key, []).append(row)
    writes = 0
    for batch in prediction_batches(root):
        previous = latest_prop_results(root, batch["run_id"])
        changed = []
        for p in batch.get("player_predictions", []):
            game = games.get(str(p["game_id"]))
            if not game or p.get("team") not in {game["home_team"], game["away_team"]}:
                continue
            candidates = lookup.get((p["season"], p["week"], p["player_id"], p["team"]), [])
            if len(candidates) != 1:
                continue
            actual = candidates[0]
            if actual.get("game_id") and str(actual["game_id"]) != p["game_id"]:
                continue
            if actual.get("opponent_team") and actual["opponent_team"] != p["opponent"]:
                continue
            value = numeric(actual.get(p["category"]))
            if value is None:
                continue
            row = {
                "prop_id": prop_id(p),
                "game_id": p["game_id"],
                "actual": value,
                "status": "final",
            }
            if any(previous.get(row["prop_id"], {}).get(k) != v for k, v in row.items()):
                changed.append(row)
        if changed:
            events = prop_events(root, batch["run_id"])
            revision = max((e["revision"] for e in events), default=0) + 1
            path = Path(root) / "prop_settlements" / batch["run_id"] / f"{revision:06d}.json"
            if path.exists():
                raise FileExistsError("Concurrent player settlement; retry")
            atomic_write_json(
                path,
                {
                    "run_id": batch["run_id"],
                    "revision": revision,
                    "scored_at": now.isoformat(),
                    "source": "nflverse weekly player stats",
                    "results": changed,
                },
            )
            writes += 1
    return writes


def player_forecast_rows(
    root: str | Path, *, policy="first", as_of: datetime | None = None
) -> pd.DataFrame:
    if policy not in {"first", "horizon"}:
        raise ValueError("Unknown forecast selection policy")
    now = as_of or datetime.now(UTC)
    records = []
    for batch in prediction_batches(root):
        published = parse_timestamp(batch["created_at"])
        if published > now:
            continue
        settled = latest_prop_results(root, batch["run_id"])
        for p in batch.get("player_predictions", []):
            kickoff = parse_timestamp(p["commence_time"])
            if published >= kickoff or (
                policy == "horizon" and published > kickoff - timedelta(minutes=60)
            ):
                continue
            actual = settled.get(prop_id(p), {})
            line = numeric(p.get("market_line"))
            try:
                market_at = parse_timestamp(p.get("market_at") or "")
                captured = parse_timestamp(p.get("market_captured_at") or "")
                valid = (
                    market_at <= captured <= published < kickoff
                    and published - market_at <= timedelta(hours=1)
                )
            except (KeyError, TypeError, ValueError):
                valid = False
            status = actual.get("status", "awaiting stats" if kickoff <= now else "scheduled")
            records.append(
                {
                    **p,
                    "market_line": line if valid else None,
                    "actual": actual.get("actual"),
                    "status": status,
                    "published_at": published,
                    "run_id": batch["run_id"],
                    "model_hash": batch["model_hash"],
                    "data_cutoff": batch["data_cutoff"],
                    "revision": actual.get("revision"),
                    "scored_at": actual.get("scored_at"),
                }
            )
    if not records:
        return pd.DataFrame()
    rows = (
        pd.DataFrame(records)
        .sort_values(["published_at", "run_id"])
        .drop_duplicates(
            ["season", "game_id", "player_id", "category"],
            keep="first" if policy == "first" else "last",
        )
    )
    rows["result"] = [
        grade_pick(r["projection"], r["market_line"], r["actual"], r["status"])
        for r in rows.to_dict("records")
    ]
    rows["error"] = (
        (rows.projection - pd.to_numeric(rows.actual, errors="coerce"))
        .abs()
        .where(rows.status.eq("final"))
    )
    return rows.reset_index(drop=True)


def player_breakdown(rows: pd.DataFrame) -> pd.DataFrame:
    """Within one prop category, compare players without mixing yards and catches."""
    from .results_tracker import pick_record

    records = []
    if rows.empty:
        return pd.DataFrame()
    for (player_id, category), group in rows.groupby(["player_id", "category"]):
        settled = group[group.status.eq("final")].dropna(subset=["actual", "projection"])
        if settled.empty:
            continue
        record = pick_record(settled.result)
        latest = group.sort_values("published_at").iloc[-1]
        records.append(
            {
                "player_id": player_id,
                "player": latest.player_name,
                "team": latest.team,
                "category": category,
                "games": len(settled),
                "mae": settled.error.mean(),
                "wins": record["wins"],
                "losses": record["losses"],
                "pushes": record["pushes"],
                "rate": record["rate"],
                "decisions": record["decisions"],
            }
        )
    return (
        pd.DataFrame(records).sort_values(["mae", "games", "player"], ascending=[True, False, True])
        if records
        else pd.DataFrame()
    )
