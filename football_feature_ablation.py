#!/usr/bin/env python3
"""Benchmark leak-safe football feature groups without requesting market data."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nfl_prediction.config import get_season_context
from nfl_prediction.data import load_nflverse_data
from nfl_prediction.features import (
    CANDIDATE_GAME_FEATURE_GROUPS,
    CORE_GAME_FEATURES,
    build_point_in_time_game_features,
)
from nfl_prediction.historical_market import prequential_component_blend
from nfl_prediction.io import atomic_write_json
from nfl_prediction.modeling import chronological_oof_predictions


def _feature_sets() -> dict[str, list[str]]:
    groups = list(CANDIDATE_GAME_FEATURE_GROUPS)
    configurations = {"core": list(CORE_GAME_FEATURES)}
    for count in range(1, len(groups) + 1):
        for selected in combinations(groups, count):
            name = "core+" + "+".join(selected)
            configurations[name] = list(CORE_GAME_FEATURES) + [
                feature for group in selected for feature in CANDIDATE_GAME_FEATURE_GROUPS[group]
            ]
    return configurations


def _evaluate(
    games: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    min_train_rows: int,
) -> dict[str, Any]:
    actual, first, second, indices = chronological_oof_predictions(
        games,
        features,
        target,
        min_train_rows=min_train_rows,
    )
    validation = games.loc[indices].copy()
    predictions, weights = prequential_component_blend(validation, actual, first, second)
    residuals = predictions - actual
    by_season: dict[str, dict[str, float | int]] = {}
    seasons = validation["season"].astype(int).to_numpy()
    for season in sorted(set(seasons)):
        mask = seasons == season
        by_season[str(season)] = {
            "rows": int(mask.sum()),
            "mae": float(mean_absolute_error(actual[mask], predictions[mask])),
        }
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
        "bias": float(np.mean(residuals)),
        "mean_ridge_weight": float(np.mean(weights)),
        "by_season": by_season,
    }


def run_ablation(seasons: list[int], *, min_train_rows: int) -> dict[str, Any]:
    data = load_nflverse_data(seasons)
    games = build_point_in_time_game_features(
        data.schedules,
        data.pbp,
        include_unplayed=False,
        rosters=data.rosters,
        snap_counts=data.snap_counts,
    ).games.dropna(subset=["home_margin", "total_points"])
    configurations = _feature_sets()
    results: dict[str, Any] = {}
    for name, features in configurations.items():
        results[name] = {
            "feature_count": len(features),
            "groups": [] if name == "core" else name.removeprefix("core+").split("+"),
            "margin": _evaluate(games, features, "home_margin", min_train_rows=min_train_rows),
            "total": _evaluate(games, features, "total_points", min_train_rows=min_train_rows),
        }

    core = results["core"]
    for result in results.values():
        for target in ("margin", "total"):
            result[target]["mae_change_vs_core"] = result[target]["mae"] - core[target]["mae"]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "nflverse play-by-play and schedules only; no Odds API calls",
        "seasons_requested": seasons,
        "completed_games": int(len(games)),
        "min_train_rows": min_train_rows,
        "methodology": {
            "validation": "expanding-window chronological out-of-fold by NFL week",
            "blending": "ridge/gradient-boosting weights learned from earlier OOF weeks only",
            "feature_timing": "all team features shifted; games use completed prior dates only",
            "comparison": "all predeclared combinations of four feature groups",
        },
        "feature_groups": CANDIDATE_GAME_FEATURE_GROUPS,
        "results": results,
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
    parser.add_argument("--min-train-rows", type=int, default=350)
    parser.add_argument("--output", type=Path, default=Path("football_feature_benchmark.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_ablation(args.seasons, min_train_rows=args.min_train_rows)
    atomic_write_json(args.output, report)
    summary = {
        name: {target: round(metrics[target]["mae"], 4) for target in ("margin", "total")}
        for name, metrics in report["results"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
