from __future__ import annotations

import json
import math
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MARKET_PRIVATE_DIR
from .data import load_nflverse_data
from .features import GAME_FEATURES, build_point_in_time_game_features
from .io import atomic_write_json, read_json
from .modeling import chronological_oof_predictions
from .odds import (
    DEFAULT_MARKETS,
    MarketSnapshotStore,
    OddsApiClient,
    build_consensus,
    estimate_historical_credits,
    parse_timestamp,
)


@dataclass(frozen=True)
class SnapshotRequest:
    requested_at: str
    kickoff_at: str
    game_indices: tuple[int, ...]


def _kickoff_utc(row: pd.Series) -> datetime:
    gameday = pd.Timestamp(row["gameday"]).strftime("%Y-%m-%d")
    gametime = str(row.get("gametime", ""))
    if not gametime or gametime == "TBD":
        raise ValueError(f"Missing kickoff time for game {row.get('game_id', row.name)}")
    local = datetime.fromisoformat(f"{gameday}T{gametime}").replace(
        tzinfo=ZoneInfo("America/New_York")
    )
    return local.astimezone(UTC)


def build_snapshot_plan(
    games: pd.DataFrame,
    *,
    minutes_before_kickoff: int = 30,
) -> list[SnapshotRequest]:
    """Group games by kickoff so one paid snapshot can cover an entire window."""
    if minutes_before_kickoff <= 0:
        raise ValueError("minutes_before_kickoff must be positive")
    groups: dict[datetime, list[int]] = {}
    for index, game in games.iterrows():
        kickoff = _kickoff_utc(game)
        groups.setdefault(kickoff, []).append(int(index))
    return [
        SnapshotRequest(
            requested_at=(kickoff - timedelta(minutes=minutes_before_kickoff)).isoformat(),
            kickoff_at=kickoff.isoformat(),
            game_indices=tuple(indices),
        )
        for kickoff, indices in sorted(groups.items())
    ]


def learn_mae_weight(
    actual: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    grid_size: int = 101,
) -> float:
    if not len(actual):
        return 0.5
    candidates = np.linspace(0.0, 1.0, grid_size)
    return float(
        min(
            candidates,
            key=lambda weight: mean_absolute_error(
                actual, weight * first + (1.0 - weight) * second
            ),
        )
    )


def prequential_component_blend(
    validation: pd.DataFrame,
    actual: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    min_meta_rows: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend each week using component residuals from earlier validation weeks only."""
    predictions = np.empty(len(actual), dtype=float)
    weights = np.empty(len(actual), dtype=float)
    prior_positions: list[int] = []
    keys = validation[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)
    seasons = validation["season"].to_numpy()
    weeks = validation["week"].to_numpy()
    for season, week in keys:
        positions = np.flatnonzero((seasons == season) & (weeks == week))
        if len(prior_positions) >= min_meta_rows:
            prior = np.asarray(prior_positions, dtype=int)
            weight = learn_mae_weight(actual[prior], first[prior], second[prior])
        else:
            weight = 0.5
        predictions[positions] = weight * first[positions] + (1.0 - weight) * second[positions]
        weights[positions] = weight
        prior_positions.extend(positions.tolist())
    return predictions, weights


def build_independent_oof_games(
    *,
    training_seasons: list[int],
    min_train_rows: int = 350,
) -> pd.DataFrame:
    data = load_nflverse_data(training_seasons)
    games = build_point_in_time_game_features(
        data.schedules, data.pbp, include_unplayed=False
    ).games
    completed = games.dropna(subset=["home_margin", "total_points"]).copy()
    outputs: list[pd.DataFrame] = []
    for target, output_name in (
        ("home_margin", "independent_home_margin"),
        ("total_points", "independent_total"),
    ):
        actual, first, second, indices = chronological_oof_predictions(
            completed,
            GAME_FEATURES,
            target,
            min_train_rows=min_train_rows,
        )
        validation = completed.loc[indices].copy()
        predictions, weights = prequential_component_blend(validation, actual, first, second)
        outputs.append(
            pd.DataFrame(
                {
                    "frame_index": indices,
                    f"actual_{target}": actual,
                    output_name: predictions,
                    f"{target}_ridge_weight": weights,
                }
            )
        )
    merged = outputs[0].merge(outputs[1], on="frame_index", validate="one_to_one")
    identifiers = completed.loc[
        merged["frame_index"],
        ["game_id", "season", "week", "gameday", "gametime", "home_team", "away_team"],
    ].reset_index(names="frame_index")
    result = identifiers.merge(merged, on="frame_index", validate="one_to_one")
    return result.sort_values(["season", "week", "gameday", "gametime"]).reset_index(drop=True)


def _request_consensus_path(request: SnapshotRequest, private_root: Path) -> Path:
    stamp = parse_timestamp(request.requested_at).strftime("%Y%m%dT%H%M%SZ")
    return private_root / "consensus" / f"benchmark-{stamp}.json"


def collect_historical_consensus(
    games: pd.DataFrame,
    requests: list[SnapshotRequest],
    *,
    client: OddsApiClient,
    regions: str = "us",
    markets: tuple[str, ...] = DEFAULT_MARKETS,
    max_credits: int,
    private_root: str | Path = MARKET_PRIVATE_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    private_path = Path(private_root)
    uncached = [
        request
        for request in requests
        if not _request_consensus_path(request, private_path).exists()
    ]
    estimated = estimate_historical_credits(
        regions=regions, markets=markets, snapshots=len(uncached)
    )
    if estimated > max_credits:
        raise ValueError(f"Estimated historical cost {estimated} exceeds hard budget {max_credits}")
    records: list[dict[str, Any]] = []
    quota: dict[str, int | None] = {}
    actual_credits = 0
    missing_games = 0
    for request in requests:
        consensus_path = _request_consensus_path(request, private_path)
        consensus = read_json(consensus_path)
        if consensus is None:
            fetch = client.historical_odds(request.requested_at, regions=regions, markets=markets)
            consensus = build_consensus(fetch, regions=regions, markets=markets)
            store = MarketSnapshotStore(
                private_root=private_path,
                consensus_path=consensus_path,
            )
            with suppress(FileExistsError):
                store.save(fetch, consensus)
            quota = asdict(fetch.credits)
            actual_credits += fetch.credits.last_request or 0
        lookup = {
            (event["away_team"], event["home_team"]): event for event in consensus.get("games", [])
        }
        for game_index in request.game_indices:
            game = games.loc[game_index]
            event = lookup.get((game["away_team"], game["home_team"]))
            if event is None or not event.get("spread") or not event.get("total"):
                missing_games += 1
                continue
            event_kickoff = parse_timestamp(event["commence_time"])
            planned_kickoff = parse_timestamp(request.kickoff_at)
            if abs((event_kickoff - planned_kickoff).total_seconds()) > 6 * 3600:
                missing_games += 1
                continue
            snapshot_at = parse_timestamp(consensus["snapshot_at"])
            if snapshot_at >= planned_kickoff:
                missing_games += 1
                continue
            spread = event["spread"]
            total = event["total"]
            records.append(
                {
                    **game.to_dict(),
                    "market_snapshot_at": snapshot_at.isoformat(),
                    "market_home_spread": float(spread["home_spread"]),
                    "market_home_margin": float(spread["market_home_margin"]),
                    "market_total": float(total["total"]),
                    "spread_book_count": int(spread["book_count"]),
                    "total_book_count": int(total["book_count"]),
                    "spread_line_iqr": spread.get("line_iqr"),
                    "total_line_iqr": total.get("line_iqr"),
                }
            )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values(["season", "week", "gameday", "gametime"])
    metadata = {
        "request_count": len(requests),
        "api_requests": len(uncached),
        "cache_hits": len(requests) - len(uncached),
        "estimated_credits": estimated,
        "actual_credits": actual_credits,
        "matched_games": len(frame),
        "missing_games": missing_games,
        "quota_after": quota,
    }
    return frame, metadata


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "bias": float(np.mean(predicted - actual)),
    }


def add_prequential_market_blends(
    frame: pd.DataFrame,
    *,
    min_meta_rows: int = 64,
) -> pd.DataFrame:
    ordered = frame.sort_values(["season", "week", "gameday", "gametime"]).copy()
    for target, independent_column, market_column in (
        ("margin", "independent_home_margin", "market_home_margin"),
        ("total", "independent_total", "market_total"),
    ):
        actual = ordered[
            f"actual_{'home_margin' if target == 'margin' else 'total_points'}"
        ].to_numpy(dtype=float)
        independent = ordered[independent_column].to_numpy(dtype=float)
        market = ordered[market_column].to_numpy(dtype=float)
        blend = np.empty(len(ordered), dtype=float)
        weights = np.empty(len(ordered), dtype=float)
        seasons = ordered["season"].to_numpy()
        weeks = ordered["week"].to_numpy()
        prior_positions: list[int] = []
        keys = ordered[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)
        for season, week in keys:
            positions = np.flatnonzero((seasons == season) & (weeks == week))
            if len(prior_positions) >= min_meta_rows:
                prior = np.asarray(prior_positions, dtype=int)
                market_weight = learn_mae_weight(actual[prior], market[prior], independent[prior])
            else:
                market_weight = 0.5
            blend[positions] = (
                market_weight * market[positions] + (1.0 - market_weight) * independent[positions]
            )
            weights[positions] = market_weight
            prior_positions.extend(positions.tolist())
        ordered[f"blended_{target}"] = blend
        ordered[f"market_{target}_weight"] = weights
    return ordered


def _disagreement_buckets(frame: pd.DataFrame) -> list[dict[str, Any]]:
    difference = frame["independent_home_margin"] - frame["market_home_margin"]
    absolute = difference.abs()
    definitions = [
        ("under_2", 0.0, 2.0),
        ("2_to_4", 2.0, 4.0),
        ("4_to_7", 4.0, 7.0),
        ("7_plus", 7.0, math.inf),
    ]
    output = []
    actual = frame["actual_home_margin"]
    ats_margin = actual - frame["market_home_margin"]
    for label, lower, upper in definitions:
        mask = absolute.ge(lower) & absolute.lt(upper)
        subset = frame[mask]
        if subset.empty:
            output.append({"bucket": label, "games": 0, "ats_decisions": 0, "ats_win_rate": None})
            continue
        model_side = np.sign(difference[mask])
        result = np.sign(ats_margin[mask]) * model_side
        decisions = result[result.ne(0)]
        output.append(
            {
                "bucket": label,
                "games": int(mask.sum()),
                "ats_decisions": int(len(decisions)),
                "ats_win_rate": float(decisions.gt(0).mean()) if len(decisions) else None,
                "independent_margin_mae": float(
                    mean_absolute_error(
                        subset["actual_home_margin"], subset["independent_home_margin"]
                    )
                ),
                "market_margin_mae": float(
                    mean_absolute_error(subset["actual_home_margin"], subset["market_home_margin"])
                ),
            }
        )
    return output


def build_aggregate_report(
    collected: pd.DataFrame,
    collection_metadata: dict[str, Any],
    *,
    training_seasons: list[int],
    evaluation_seasons: list[int],
    minutes_before_kickoff: int,
) -> dict[str, Any]:
    if collected.empty:
        raise ValueError("No historical games matched the requested market snapshots")
    frame = add_prequential_market_blends(collected)
    margin_actual = frame["actual_home_margin"].to_numpy(dtype=float)
    total_actual = frame["actual_total_points"].to_numpy(dtype=float)
    variants = {
        "independent_margin": _regression_metrics(
            margin_actual, frame["independent_home_margin"].to_numpy(dtype=float)
        ),
        "market_margin": _regression_metrics(
            margin_actual, frame["market_home_margin"].to_numpy(dtype=float)
        ),
        "walk_forward_blended_margin": _regression_metrics(
            margin_actual, frame["blended_margin"].to_numpy(dtype=float)
        ),
        "independent_total": _regression_metrics(
            total_actual, frame["independent_total"].to_numpy(dtype=float)
        ),
        "market_total": _regression_metrics(
            total_actual, frame["market_total"].to_numpy(dtype=float)
        ),
        "walk_forward_blended_total": _regression_metrics(
            total_actual, frame["blended_total"].to_numpy(dtype=float)
        ),
    }
    by_season = []
    for season, subset in frame.groupby("season"):
        by_season.append(
            {
                "season": int(season),
                "games": int(len(subset)),
                "independent_margin_mae": float(
                    mean_absolute_error(
                        subset["actual_home_margin"], subset["independent_home_margin"]
                    )
                ),
                "market_margin_mae": float(
                    mean_absolute_error(subset["actual_home_margin"], subset["market_home_margin"])
                ),
                "blended_margin_mae": float(
                    mean_absolute_error(subset["actual_home_margin"], subset["blended_margin"])
                ),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": "The Odds API",
        "methodology": {
            "training_seasons": training_seasons,
            "evaluation_seasons": evaluation_seasons,
            "snapshot_minutes_before_kickoff": minutes_before_kickoff,
            "independent_predictions": "weekly expanding-window OOF; component weights use prior OOF weeks only",
            "market_blend": "MAE-optimal convex weight learned from prior matched weeks only",
            "raw_market_data_published": False,
        },
        "collection": collection_metadata,
        "games": int(len(frame)),
        "average_spread_books": float(frame["spread_book_count"].mean()),
        "average_total_books": float(frame["total_book_count"].mean()),
        "variants": variants,
        "latest_market_margin_weight": float(frame["market_margin_weight"].iloc[-1]),
        "latest_market_total_weight": float(frame["market_total_weight"].iloc[-1]),
        "by_season": by_season,
        "disagreement_buckets": _disagreement_buckets(frame),
    }


def save_private_records(frame: pd.DataFrame) -> Path:
    target = MARKET_PRIVATE_DIR / "benchmark_records.json"
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    atomic_write_json(target, records)
    return target
