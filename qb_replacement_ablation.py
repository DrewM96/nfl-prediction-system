#!/usr/bin/env python3
"""Backtest research-only NFL quarterback replacement-value features."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from nfl_prediction.config import get_season_context
from nfl_prediction.data import load_nflverse_data
from nfl_prediction.features import (
    GAME_MARGIN_FEATURES,
    GAME_TOTAL_FEATURES,
    build_point_in_time_game_features,
)
from nfl_prediction.historical_market import prequential_component_blend
from nfl_prediction.injuries import filter_to_injury_covered_games
from nfl_prediction.io import atomic_write_json
from nfl_prediction.modeling import GAME_RIDGE_ALPHA, chronological_oof_predictions
from nfl_prediction.qb_replacement import (
    QB_REPLACEMENT_CANDIDATE_GROUPS,
    attach_qb_replacement_features,
    build_qb_replacement_table,
)


def _feature_sets(base_features: list[str]) -> dict[str, list[str]]:
    configs = {"production_base": list(base_features)}
    for group in ("qb_availability", "qb_value_gap", "qb_expected_points_loss"):
        configs[f"production_base+{group}"] = (
            list(base_features) + QB_REPLACEMENT_CANDIDATE_GROUPS[group]
        )
    configs["production_base+qb_availability+qb_expected_points_loss"] = (
        list(base_features)
        + QB_REPLACEMENT_CANDIDATE_GROUPS["qb_availability"]
        + QB_REPLACEMENT_CANDIDATE_GROUPS["qb_expected_points_loss"]
    )
    return configs


def _split_metrics(
    actual: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    if not mask.any():
        return {"rows": 0, "mae": None}
    return {
        "rows": int(mask.sum()),
        "mae": float(mean_absolute_error(actual[mask], predictions[mask])),
    }


def _evaluate(
    games: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    min_train_rows: int,
    holdout_season: int,
) -> dict[str, Any]:
    actual, first, second, indices = chronological_oof_predictions(
        games,
        features,
        target,
        min_train_rows=min_train_rows,
        ridge_alpha=GAME_RIDGE_ALPHA,
    )
    validation = games.loc[indices].copy()
    predictions, weights = prequential_component_blend(validation, actual, first, second)
    seasons = validation["season"].astype(int).to_numpy()
    return {
        "all_oof": _split_metrics(actual, predictions, np.ones(len(actual), dtype=bool)),
        "development": _split_metrics(actual, predictions, seasons < holdout_season),
        "holdout": _split_metrics(actual, predictions, seasons == holdout_season),
        "by_season": {
            str(season): _split_metrics(actual, predictions, seasons == season)
            for season in sorted(set(seasons))
        },
        "mean_ridge_weight": float(np.mean(weights)),
    }


def _evaluate_target(
    games: pd.DataFrame,
    *,
    base_features: list[str],
    target: str,
    min_train_rows: int,
    holdout_season: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, features in _feature_sets(base_features).items():
        results[name] = {
            "feature_count": len(features),
            "groups": (
                []
                if name == "production_base"
                else name.removeprefix("production_base+").split("+")
            ),
            "metrics": _evaluate(
                games,
                features,
                target,
                min_train_rows=min_train_rows,
                holdout_season=holdout_season,
            ),
        }

    base = results["production_base"]["metrics"]
    for result in results.values():
        for split in ("all_oof", "development", "holdout"):
            current = result["metrics"][split]["mae"]
            baseline = base[split]["mae"]
            result["metrics"][split]["mae_change_vs_base"] = (
                current - baseline
                if current is not None and baseline is not None
                else None
            )
    return results


def _qb_event_summary(qb_table: pd.DataFrame) -> dict[str, Any]:
    if qb_table.empty:
        return {
            "team_weeks": 0,
            "starter_reported": 0,
            "starter_unavailable_weight_positive": 0,
            "mean_expected_points_lost_when_positive": None,
            "p90_expected_points_lost_when_positive": None,
            "max_expected_points_lost": None,
        }
    positive = qb_table[qb_table["qb_unavailability_weight"].gt(0)]
    losses = positive["qb_expected_points_lost"]
    return {
        "team_weeks": int(len(qb_table)),
        "starter_reported": int(qb_table["qb_starter_reported"].sum()),
        "starter_unavailable_weight_positive": int(len(positive)),
        "mean_expected_points_lost_when_positive": (
            float(losses.mean()) if not losses.empty else None
        ),
        "p90_expected_points_lost_when_positive": (
            float(losses.quantile(0.90)) if not losses.empty else None
        ),
        "max_expected_points_lost": (
            float(losses.max()) if not losses.empty else None
        ),
    }


def run_ablation(
    seasons: list[int],
    *,
    min_train_rows: int,
    holdout_season: int,
    lookback_weeks: int,
    shrinkage_dropbacks: float,
) -> dict[str, Any]:
    data = load_nflverse_data(seasons)
    game_result = build_point_in_time_game_features(
        data.schedules,
        data.pbp,
        include_unplayed=False,
        rosters=data.rosters,
        snap_counts=data.snap_counts,
    )
    games = game_result.games.dropna(subset=["home_margin", "total_points"]).copy()
    qb_table = build_qb_replacement_table(
        data.injuries,
        data.pbp,
        data.rosters,
        lookback_weeks=lookback_weeks,
        shrinkage_dropbacks=shrinkage_dropbacks,
    )
    games = attach_qb_replacement_features(games, qb_table)
    games = filter_to_injury_covered_games(games, data.injuries)
    if games.empty:
        raise RuntimeError("No completed games overlap the nflverse injury feed.")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "nflverse injury reports and play-by-play; no Odds API calls",
        "seasons_requested": seasons,
        "holdout_season": holdout_season,
        "lookback_weeks": lookback_weeks,
        "shrinkage_dropbacks": shrinkage_dropbacks,
        "completed_games_with_injury_feed": int(len(games)),
        "qb_event_summary": _qb_event_summary(qb_table),
        "methodology": {
            "validation": "expanding-window chronological out-of-fold by NFL week",
            "base_models": (
                "current production feature sets: margin uses GAME_MARGIN_FEATURES; "
                "total uses GAME_TOTAL_FEATURES"
            ),
            "starter_detection": (
                "prior team dropback leader across the previous four team games; "
                "prior-season same-team fallback when the current season has no history"
            ),
            "qb_value": (
                "EPA/dropback shrunk toward a league prior with 80 equivalent dropbacks"
            ),
            "replacement": (
                "next-highest prior team dropback quarterback; league prior when no "
                "backup history exists"
            ),
            "expected_points_loss": (
                "injury severity x (starter EPA/dropback - backup EPA/dropback) x "
                "expected team dropbacks"
            ),
            "timing_limit": (
                "historical nflverse injury rows are a late-week/final-report proxy, "
                "not a timestamped reconstruction of Tuesday information"
            ),
            "candidate_policy": (
                "predeclared focused tests: availability, value gap, expected points loss, "
                "and availability plus expected points loss; redundant combinations are omitted"
            ),
            "promotion_policy": (
                "no production promotion unless development and separate holdout both improve "
                "and prospective timestamp-compatible validation agrees"
            ),
        },
        "feature_groups": QB_REPLACEMENT_CANDIDATE_GROUPS,
        "margin": _evaluate_target(
            games,
            base_features=GAME_MARGIN_FEATURES,
            target="home_margin",
            min_train_rows=min_train_rows,
            holdout_season=holdout_season,
        ),
        "total": _evaluate_target(
            games,
            base_features=GAME_TOTAL_FEATURES,
            target="total_points",
            min_train_rows=min_train_rows,
            holdout_season=holdout_season,
        ),
    }


def parse_args() -> argparse.Namespace:
    context = get_season_context()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(context.training_seasons),
    )
    parser.add_argument("--holdout-season", type=int, default=context.prediction_season - 1)
    parser.add_argument("--min-train-rows", type=int, default=350)
    parser.add_argument("--lookback-weeks", type=int, default=4)
    parser.add_argument("--shrinkage-dropbacks", type=float, default=80.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qb_replacement_benchmark.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_ablation(
        args.seasons,
        min_train_rows=args.min_train_rows,
        holdout_season=args.holdout_season,
        lookback_weeks=args.lookback_weeks,
        shrinkage_dropbacks=args.shrinkage_dropbacks,
    )
    atomic_write_json(args.output, report)
    summary = {
        target: {
            name: {
                split: (
                    round(config["metrics"][split]["mae"], 4)
                    if config["metrics"][split]["mae"] is not None
                    else None
                )
                for split in ("development", "holdout", "all_oof")
            }
            for name, config in report[target].items()
        }
        for target in ("margin", "total")
    }
    print(json.dumps({"qb_events": report["qb_event_summary"], "results": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
