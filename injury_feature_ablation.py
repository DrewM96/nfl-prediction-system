#!/usr/bin/env python3
"""Backtest research-only NFL injury availability features without paid data."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from itertools import combinations
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
from nfl_prediction.injuries import (
    INJURY_CANDIDATE_GAME_FEATURE_GROUPS,
    attach_injury_availability_features,
    build_injury_availability_table,
    filter_to_injury_covered_games,
)
from nfl_prediction.io import atomic_write_json
from nfl_prediction.modeling import GAME_RIDGE_ALPHA, chronological_oof_predictions


def _feature_sets(base_features: list[str]) -> dict[str, list[str]]:
    groups = list(INJURY_CANDIDATE_GAME_FEATURE_GROUPS)
    configurations = {"production_base": list(base_features)}
    for count in range(1, len(groups) + 1):
        for selected in combinations(groups, count):
            name = "production_base+" + "+".join(selected)
            configurations[name] = list(base_features) + [
                feature
                for group in selected
                for feature in INJURY_CANDIDATE_GAME_FEATURE_GROUPS[group]
            ]
    return configurations


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
    by_season = {
        str(season): _split_metrics(actual, predictions, seasons == season)
        for season in sorted(set(seasons))
    }
    return {
        "all_oof": _split_metrics(actual, predictions, np.ones(len(actual), dtype=bool)),
        "development": _split_metrics(actual, predictions, seasons < holdout_season),
        "holdout": _split_metrics(actual, predictions, seasons == holdout_season),
        "by_season": by_season,
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
                current - baseline if current is not None and baseline is not None else None
            )
    return results


def run_ablation(
    seasons: list[int],
    *,
    min_train_rows: int,
    holdout_season: int,
    lookback_weeks: int,
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
    availability = build_injury_availability_table(
        data.injuries,
        data.snap_counts,
        data.rosters,
        lookback_weeks=lookback_weeks,
    )
    games = attach_injury_availability_features(games, availability)
    games = filter_to_injury_covered_games(games, data.injuries)

    if games.empty:
        raise RuntimeError("No completed games overlap the nflverse injury feed.")

    margin = _evaluate_target(
        games,
        base_features=GAME_MARGIN_FEATURES,
        target="home_margin",
        min_train_rows=min_train_rows,
        holdout_season=holdout_season,
    )
    total = _evaluate_target(
        games,
        base_features=GAME_TOTAL_FEATURES,
        target="total_points",
        min_train_rows=min_train_rows,
        holdout_season=holdout_season,
    )
    coverage = sorted(
        {
            (int(row.season), int(row.week))
            for row in games[["season", "week"]].drop_duplicates().itertuples(index=False)
        }
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "nflverse injury reports, weekly rosters, and snap counts; no Odds API calls",
        "seasons_requested": seasons,
        "holdout_season": holdout_season,
        "lookback_weeks": lookback_weeks,
        "completed_games_with_injury_feed": int(len(games)),
        "injury_team_week_rows": int(len(availability)),
        "covered_season_weeks": [{"season": season, "week": week} for season, week in coverage],
        "methodology": {
            "validation": "expanding-window chronological out-of-fold by NFL week",
            "blending": "ridge/gradient-boosting weights learned from earlier OOF weeks only",
            "base_models": (
                "current production feature sets: margin uses GAME_MARGIN_FEATURES; "
                "total uses GAME_TOTAL_FEATURES"
            ),
            "injury_timing": (
                "historical nflverse weekly/final injury report proxy; suitable for late-week "
                "research, not evidence of what an earlier Tuesday forecast knew"
            ),
            "player_weight": (
                f"prior {lookback_weeks} team games of offensive/defensive snap share; "
                "prior-season same-team fallback when current-season history is unavailable"
            ),
            "status_weights": (
                "Out=1.00, Doubtful=0.75, Questionable=0.35; if no game status, "
                "DNP practice=0.25, Limited=0.10, Full=0.00"
            ),
            "selection_policy": (
                "candidate groups are research-only; no production promotion without "
                "holdout improvement and timestamp-compatible prospective validation"
            ),
        },
        "feature_groups": INJURY_CANDIDATE_GAME_FEATURE_GROUPS,
        "margin": margin,
        "total": total,
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
    parser.add_argument("--output", type=Path, default=Path("injury_feature_benchmark.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_ablation(
        args.seasons,
        min_train_rows=args.min_train_rows,
        holdout_season=args.holdout_season,
        lookback_weeks=args.lookback_weeks,
    )
    atomic_write_json(args.output, report)
    summary = {}
    for target in ("margin", "total"):
        summary[target] = {
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
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
