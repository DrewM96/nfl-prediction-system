#!/usr/bin/env python3
"""Compare predeclared model profiles on leak-safe football features."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_prediction.config import get_season_context
from nfl_prediction.data import load_nflverse_data
from nfl_prediction.features import CORE_GAME_FEATURES, build_point_in_time_game_features
from nfl_prediction.historical_market import prequential_component_blend
from nfl_prediction.io import atomic_write_json
from nfl_prediction.modeling import chronological_oof_predictions


def _ridge(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def _profiles() -> dict[str, list[Any]]:
    def current_tree() -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.035,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            loss="huber",
        )

    return {
        "current": [
            _ridge(10.0),
            current_tree(),
        ],
        "ridge30_current_tree": [
            _ridge(30.0),
            current_tree(),
        ],
        "ridge50_current_tree": [
            _ridge(50.0),
            current_tree(),
        ],
        "ridge100_current_tree": [
            _ridge(100.0),
            current_tree(),
        ],
        "shallow_huber": [
            _ridge(30.0),
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=1,
                learning_rate=0.03,
                min_samples_leaf=15,
                subsample=0.8,
                random_state=42,
                loss="huber",
            ),
        ],
        "conservative_huber": [
            _ridge(50.0),
            GradientBoostingRegressor(
                n_estimators=150,
                max_depth=2,
                learning_rate=0.025,
                min_samples_leaf=20,
                subsample=0.8,
                random_state=42,
                loss="huber",
            ),
        ],
        "shallow_squared": [
            _ridge(30.0),
            GradientBoostingRegressor(
                n_estimators=150,
                max_depth=1,
                learning_rate=0.03,
                min_samples_leaf=15,
                subsample=0.8,
                random_state=42,
                loss="squared_error",
            ),
        ],
        "histogram_absolute": [
            _ridge(50.0),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=7,
                min_samples_leaf=20,
                l2_regularization=5.0,
                random_state=42,
            ),
        ],
    }


def _metrics(
    games: pd.DataFrame,
    target: str,
    models: list[Any],
    *,
    min_train_rows: int,
    holdout_season: int,
) -> dict[str, Any]:
    actual, first, second, indices = chronological_oof_predictions(
        games,
        CORE_GAME_FEATURES,
        target,
        min_train_rows=min_train_rows,
        candidate_models=models,
    )
    validation = games.loc[indices].copy()
    predictions, weights = prequential_component_blend(validation, actual, first, second)
    seasons = validation["season"].astype(int).to_numpy()

    def split_metrics(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "rows": int(mask.sum()),
            "mae": float(mean_absolute_error(actual[mask], predictions[mask])),
            "ridge_mae": float(mean_absolute_error(actual[mask], first[mask])),
            "tree_mae": float(mean_absolute_error(actual[mask], second[mask])),
            "mean_ridge_weight": float(np.mean(weights[mask])),
        }

    development = seasons < holdout_season
    holdout = seasons == holdout_season
    return {
        "development": split_metrics(development),
        "holdout": split_metrics(holdout),
        "all_oof": split_metrics(np.ones(len(actual), dtype=bool)),
    }


def run_tuning(seasons: list[int], *, min_train_rows: int, holdout_season: int) -> dict[str, Any]:
    data = load_nflverse_data(seasons)
    games = build_point_in_time_game_features(
        data.schedules, data.pbp, include_unplayed=False
    ).games.dropna(subset=["home_margin", "total_points"])
    results = {
        name: {
            target: _metrics(
                games,
                target_name,
                models,
                min_train_rows=min_train_rows,
                holdout_season=holdout_season,
            )
            for target, target_name in (
                ("margin", "home_margin"),
                ("total", "total_points"),
            )
        }
        for name, models in _profiles().items()
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "nflverse play-by-play and schedules only; no Odds API calls",
        "seasons_requested": seasons,
        "holdout_season": holdout_season,
        "min_train_rows": min_train_rows,
        "methodology": {
            "selection": f"compare profiles on seasons before {holdout_season}",
            "confirmation": f"report {holdout_season} separately as a final holdout",
            "validation": "expanding-window chronological out-of-fold by NFL week",
            "blending": "ridge/tree weights learned from earlier OOF weeks only",
            "scope": "predeclared tree profiles plus a follow-up isolation of Ridge shrinkage",
        },
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
    parser.add_argument("--holdout-season", type=int, default=context.prediction_season - 1)
    parser.add_argument("--min-train-rows", type=int, default=350)
    parser.add_argument("--output", type=Path, default=Path("football_model_benchmark.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_tuning(
        args.seasons,
        min_train_rows=args.min_train_rows,
        holdout_season=args.holdout_season,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
