#!/usr/bin/env python3
"""Backtest season-opening roster-transition features without paid data."""

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
from nfl_prediction.features import CORE_GAME_FEATURES, build_point_in_time_game_features
from nfl_prediction.historical_market import prequential_component_blend
from nfl_prediction.io import atomic_write_json
from nfl_prediction.modeling import GAME_RIDGE_ALPHA, chronological_oof_predictions
from nfl_prediction.roster import ROSTER_CANDIDATE_GAME_FEATURE_GROUPS


def _feature_sets() -> dict[str, list[str]]:
    groups = list(ROSTER_CANDIDATE_GAME_FEATURE_GROUPS)
    configurations = {"core": list(CORE_GAME_FEATURES)}
    for count in range(1, len(groups) + 1):
        for selected in combinations(groups, count):
            name = "core+" + "+".join(selected)
            configurations[name] = list(CORE_GAME_FEATURES) + [
                feature
                for group in selected
                for feature in ROSTER_CANDIDATE_GAME_FEATURE_GROUPS[group]
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
    weeks = validation["week"].astype(int).to_numpy()
    early = weeks <= 4
    by_season = {
        str(season): _split_metrics(actual, predictions, early & (seasons == season))
        for season in sorted(set(seasons))
    }
    return {
        "all_oof": _split_metrics(actual, predictions, np.ones(len(actual), dtype=bool)),
        "weeks_1_4": _split_metrics(actual, predictions, early),
        "development_weeks_1_4": _split_metrics(
            actual, predictions, early & (seasons < holdout_season)
        ),
        "holdout_weeks_1_4": _split_metrics(
            actual, predictions, early & (seasons == holdout_season)
        ),
        "weeks_1_4_by_season": by_season,
        "mean_ridge_weight": float(np.mean(weights)),
    }


def run_ablation(seasons: list[int], *, min_train_rows: int, holdout_season: int) -> dict[str, Any]:
    data = load_nflverse_data(seasons)
    games = build_point_in_time_game_features(
        data.schedules,
        data.pbp,
        include_unplayed=False,
        rosters=data.rosters,
        snap_counts=data.snap_counts,
    ).games.dropna(subset=["home_margin", "total_points"])
    results: dict[str, Any] = {}
    for name, features in _feature_sets().items():
        results[name] = {
            "feature_count": len(features),
            "groups": [] if name == "core" else name.removeprefix("core+").split("+"),
            "margin": _evaluate(
                games,
                features,
                "home_margin",
                min_train_rows=min_train_rows,
                holdout_season=holdout_season,
            ),
            "total": _evaluate(
                games,
                features,
                "total_points",
                min_train_rows=min_train_rows,
                holdout_season=holdout_season,
            ),
        }

    core = results["core"]
    for result in results.values():
        for target in ("margin", "total"):
            for split in (
                "all_oof",
                "weeks_1_4",
                "development_weeks_1_4",
                "holdout_weeks_1_4",
            ):
                result[target][split]["mae_change_vs_core"] = (
                    result[target][split]["mae"] - core[target][split]["mae"]
                )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "nflverse rosters and snap counts; no Odds API calls",
        "seasons_requested": seasons,
        "holdout_season": holdout_season,
        "completed_games": int(len(games)),
        "min_train_rows": min_train_rows,
        "methodology": {
            "validation": "expanding-window chronological out-of-fold by NFL week",
            "selection_window": "Weeks 1-4, when roster features have non-zero weight",
            "confirmation": f"{holdout_season} is reported separately",
            "roster_timing": "opening roster membership; game-day active/inactive status ignored",
            "player_weight": "prior regular-season offensive or defensive snaps",
            "decay": "100%, 75%, 50%, 25%, then zero after Week 4",
            "comparison": "all predeclared combinations of four roster feature groups",
        },
        "feature_groups": ROSTER_CANDIDATE_GAME_FEATURE_GROUPS,
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
    parser.add_argument("--output", type=Path, default=Path("roster_transition_benchmark.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_ablation(
        args.seasons,
        min_train_rows=args.min_train_rows,
        holdout_season=args.holdout_season,
    )
    atomic_write_json(args.output, report)
    summary = {
        name: {
            target: {
                split: (
                    round(metrics[target][split]["mae"], 4)
                    if metrics[target][split]["mae"] is not None
                    else None
                )
                for split in ("weeks_1_4", "holdout_weeks_1_4", "all_oof")
            }
            for target in ("margin", "total")
        }
        for name, metrics in report["results"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
