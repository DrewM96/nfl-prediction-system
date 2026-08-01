#!/usr/bin/env python3
"""Run a chronological CFB football-feature benchmark without publishing raw CFBD data."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from cfb_prediction.client import CFBDClient
from cfb_prediction.features import CFB_FEATURE_CONFIGURATIONS, build_point_in_time_features
from cfb_prediction.historical import load_historical_data
from cfb_prediction.modeling import (
    chronological_ridge_predictions as _chronological_ridge_predictions,
)
from nfl_prediction.io import atomic_write_json


def _metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = mask & np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return {"rows": 0, "mae": None, "rmse": None, "bias": None}
    errors = predicted[valid] - actual[valid]
    return {
        "rows": int(valid.sum()),
        "mae": float(mean_absolute_error(actual[valid], predicted[valid])),
        "rmse": float(mean_squared_error(actual[valid], predicted[valid]) ** 0.5),
        "bias": float(np.mean(errors)),
    }


def _evaluate(
    games: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    holdout_season: int,
    min_train_rows: int,
    alpha: float,
) -> dict[str, Any]:
    actual, predicted, indices = _chronological_ridge_predictions(
        games,
        features,
        target,
        min_train_rows=min_train_rows,
        alpha=alpha,
    )
    validation = games.loc[indices]
    seasons = validation["season"].astype(int).to_numpy()
    all_rows = np.ones(len(actual), dtype=bool)
    development = seasons < holdout_season
    holdout = seasons == holdout_season
    baseline_column = "elo_expected_margin" if target == "home_margin" else "form_expected_total"
    baseline = validation[baseline_column].astype(float).to_numpy()
    market_column = "market_home_margin" if target == "home_margin" else "market_total"
    market = validation[market_column].astype(float).to_numpy()
    return {
        "model": {
            "all_oof": _metrics(actual, predicted, all_rows),
            "development": _metrics(actual, predicted, development),
            "holdout": _metrics(actual, predicted, holdout),
        },
        "football_baseline": {
            "all_oof": _metrics(actual, baseline, all_rows),
            "development": _metrics(actual, baseline, development),
            "holdout": _metrics(actual, baseline, holdout),
        },
        "listed_market": {
            "all_oof": _metrics(actual, market, all_rows),
            "development": _metrics(actual, market, development),
            "holdout": _metrics(actual, market, holdout),
        },
        "by_season": {
            str(season): _metrics(actual, predicted, seasons == season)
            for season in sorted(set(seasons))
        },
    }


def run_benchmark(
    seasons: list[int],
    *,
    holdout_season: int,
    min_train_rows: int,
    alpha: float,
    refresh: bool = False,
) -> dict[str, Any]:
    client = CFBDClient.from_environment()
    data = load_historical_data(client, seasons, refresh=refresh)
    games = build_point_in_time_features(data)
    results = {
        name: {
            "feature_count": len(features),
            "margin": _evaluate(
                games,
                features,
                "home_margin",
                holdout_season=holdout_season,
                min_train_rows=min_train_rows,
                alpha=alpha,
            ),
            "total": _evaluate(
                games,
                features,
                "total_points",
                holdout_season=holdout_season,
                min_train_rows=min_train_rows,
                alpha=alpha,
            ),
        }
        for name, features in CFB_FEATURE_CONFIGURATIONS.items()
    }
    for result in results.values():
        for target in ("margin", "total"):
            for split in ("all_oof", "development", "holdout"):
                model_mae = result[target]["model"][split]["mae"]
                baseline_mae = result[target]["football_baseline"][split]["mae"]
                result[target]["model"][split]["mae_change_vs_football_baseline"] = (
                    model_mae - baseline_mae
                    if model_mae is not None and baseline_mae is not None
                    else None
                )
    selected = {
        target: min(
            results,
            key=lambda name: results[name][target]["model"]["development"]["mae"],
        )
        for target in ("margin", "total")
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "CollegeFootballData REST API v2; raw responses remain private",
        "seasons": seasons,
        "holdout_season": holdout_season,
        "completed_fbs_games": int(len(games)),
        "min_train_rows": min_train_rows,
        "ridge_alpha": alpha,
        "methodology": {
            "validation": "features use earlier kickoffs; expanding model folds train on earlier weeks",
            "selection": f"lowest development MAE before the untouched {holdout_season} season",
            "market_role": "listed historical lines are comparison-only and never model inputs",
            "market_timing_limit": "CFBD lines lack snapshot timestamps and are not labeled closing",
            "opponent_adjustment": "pregame Elo, average opponent Elo, and margin residual versus Elo",
            "portal_timing": "records must fall within 400 days before that season's first kickoff",
            "preseason_timing_limit": "returning, talent, and recruiting feeds are season-level rather than timestamped snapshots",
            "raw_data_published": False,
        },
        "data_audit_by_season": {
            str(season): {
                "completed_fbs_games": int(games["season"].eq(season).sum()),
                "advanced_team_game_rows": int(data.advanced["season"].eq(season).sum())
                if "season" in data.advanced
                else 0,
                "returning_teams": int(
                    data.returning[data.returning["season"].eq(season)]["team"].nunique()
                )
                if {"season", "team"}.issubset(data.returning)
                else 0,
                "portal_records": int(data.portal["season"].eq(season).sum())
                if "season" in data.portal
                else 0,
                "talent_teams": int(data.talent[data.talent["season"].eq(season)]["team"].nunique())
                if {"season", "team"}.issubset(data.talent)
                else 0,
                "recruiting_teams": int(
                    data.recruiting[data.recruiting["season"].eq(season)]["team"].nunique()
                )
                if {"season", "team"}.issubset(data.recruiting)
                else 0,
                "listed_line_games": int(data.lines["season"].eq(season).sum())
                if "season" in data.lines
                else 0,
            }
            for season in seasons
        },
        "feature_configurations": CFB_FEATURE_CONFIGURATIONS,
        "selected_by_development": selected,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(range(2018, 2026)),
    )
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--min-train-rows", type=int, default=1200)
    parser.add_argument("--ridge-alpha", type=float, default=50.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cfb/historical_benchmark.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_benchmark(
        args.seasons,
        holdout_season=args.holdout_season,
        min_train_rows=args.min_train_rows,
        alpha=args.ridge_alpha,
        refresh=args.refresh,
    )
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "completed_fbs_games": report["completed_fbs_games"],
                "selected": report["selected_by_development"],
                "selected_results": {
                    target: report["results"][name][target]
                    for target, name in report["selected_by_development"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
