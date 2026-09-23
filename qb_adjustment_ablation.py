#!/usr/bin/env python3
"""Test a leak-safe post-model QB availability adjustment layer."""

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
    attach_qb_replacement_features,
    build_qb_replacement_table,
)

LAMBDA_GRID = np.linspace(0.0, 1.5, 31)


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


def _base_oof(
    games: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    min_train_rows: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    actual, first, second, indices = chronological_oof_predictions(
        games,
        features,
        target,
        min_train_rows=min_train_rows,
        ridge_alpha=GAME_RIDGE_ALPHA,
    )
    validation = games.loc[indices].copy()
    baseline, _ = prequential_component_blend(validation, actual, first, second)
    return validation, actual, baseline


def _prequential_adjustment(
    validation: pd.DataFrame,
    actual: np.ndarray,
    baseline: np.ndarray,
    raw_adjustment: np.ndarray,
    *,
    min_prior_rows: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = validation[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)
    adjusted = baseline.copy()
    lambdas = np.zeros(len(actual), dtype=float)
    seasons = validation["season"].astype(int).to_numpy()
    weeks = validation["week"].astype(int).to_numpy()

    for season, week in ordered:
        current = (seasons == int(season)) & (weeks == int(week))
        prior = (seasons < int(season)) | ((seasons == int(season)) & (weeks < int(week)))
        if int(prior.sum()) < min_prior_rows:
            weight = 0.0
        else:
            weight = min(
                LAMBDA_GRID,
                key=lambda candidate: mean_absolute_error(
                    actual[prior],
                    baseline[prior] + float(candidate) * raw_adjustment[prior],
                ),
            )
        adjusted[current] = baseline[current] + float(weight) * raw_adjustment[current]
        lambdas[current] = float(weight)
    return adjusted, lambdas


def _target_report(
    games: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    adjustment_kind: str,
    min_train_rows: int,
    holdout_season: int,
) -> dict[str, Any]:
    validation, actual, baseline = _base_oof(
        games,
        features,
        target,
        min_train_rows=min_train_rows,
    )
    home_loss = validation["home_qb_expected_points_lost"].to_numpy(dtype=float)
    away_loss = validation["away_qb_expected_points_lost"].to_numpy(dtype=float)
    if adjustment_kind == "margin":
        raw_adjustment = away_loss - home_loss
    elif adjustment_kind == "total":
        raw_adjustment = -(home_loss + away_loss)
    else:
        raise ValueError(f"Unknown adjustment kind: {adjustment_kind}")

    adjusted, lambdas = _prequential_adjustment(
        validation,
        actual,
        baseline,
        raw_adjustment,
    )
    seasons = validation["season"].astype(int).to_numpy()
    event_mask = np.abs(raw_adjustment) > 1e-12

    result = {
        "all_oof": {
            "baseline": _split_metrics(actual, baseline, np.ones(len(actual), dtype=bool)),
            "adjusted": _split_metrics(actual, adjusted, np.ones(len(actual), dtype=bool)),
        },
        "development": {
            "baseline": _split_metrics(actual, baseline, seasons < holdout_season),
            "adjusted": _split_metrics(actual, adjusted, seasons < holdout_season),
        },
        "holdout": {
            "baseline": _split_metrics(actual, baseline, seasons == holdout_season),
            "adjusted": _split_metrics(actual, adjusted, seasons == holdout_season),
        },
        "qb_event_rows": {
            "rows": int(event_mask.sum()),
            "baseline_mae": (
                float(mean_absolute_error(actual[event_mask], baseline[event_mask]))
                if event_mask.any()
                else None
            ),
            "adjusted_mae": (
                float(mean_absolute_error(actual[event_mask], adjusted[event_mask]))
                if event_mask.any()
                else None
            ),
        },
        "mean_lambda": float(np.mean(lambdas)),
        "latest_lambda": float(lambdas[-1]) if len(lambdas) else 0.0,
        "nonzero_lambda_rows": int(np.count_nonzero(lambdas)),
        "by_season": {},
    }
    for split in ("all_oof", "development", "holdout"):
        base = result[split]["baseline"]["mae"]
        changed = result[split]["adjusted"]["mae"]
        result[split]["mae_change_vs_base"] = (
            changed - base if changed is not None and base is not None else None
        )
    for season in sorted(set(seasons)):
        mask = seasons == season
        base = _split_metrics(actual, baseline, mask)
        changed = _split_metrics(actual, adjusted, mask)
        result["by_season"][str(season)] = {
            "baseline": base,
            "adjusted": changed,
            "mae_change_vs_base": changed["mae"] - base["mae"],
        }
    return result


def run_benchmark(
    seasons: list[int],
    *,
    min_train_rows: int,
    holdout_season: int,
    lookback_weeks: int,
    shrinkage_dropbacks: float,
) -> dict[str, Any]:
    data = load_nflverse_data(seasons)
    games = (
        build_point_in_time_game_features(
            data.schedules,
            data.pbp,
            include_unplayed=False,
            rosters=data.rosters,
            snap_counts=data.snap_counts,
        )
        .games.dropna(subset=["home_margin", "total_points"])
        .copy()
    )
    qb_table = build_qb_replacement_table(
        data.injuries,
        data.pbp,
        data.rosters,
        lookback_weeks=lookback_weeks,
        shrinkage_dropbacks=shrinkage_dropbacks,
    )
    games = attach_qb_replacement_features(games, qb_table)
    games = filter_to_injury_covered_games(games, data.injuries)
    positive = qb_table[qb_table["qb_expected_points_lost"].gt(0)]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "nflverse injury reports, rosters, and play-by-play",
        "seasons_requested": seasons,
        "holdout_season": holdout_season,
        "completed_games_with_injury_feed": int(len(games)),
        "qb_positive_loss_team_weeks": int(len(positive)),
        "qb_loss_summary": {
            "mean": float(positive["qb_expected_points_lost"].mean())
            if not positive.empty
            else None,
            "median": float(positive["qb_expected_points_lost"].median())
            if not positive.empty
            else None,
            "p90": float(positive["qb_expected_points_lost"].quantile(0.9))
            if not positive.empty
            else None,
            "max": float(positive["qb_expected_points_lost"].max()) if not positive.empty else None,
        },
        "methodology": {
            "architecture": "post-model availability adjustment; core football model is unchanged",
            "starter_detection": (
                "prior team dropback leader constrained to current-roster QBs; "
                "Week 1 can fall back to prior league usage for current-roster veterans"
            ),
            "qb_value": (
                "prior EPA/dropback shrunk toward league average with "
                f"{shrinkage_dropbacks:g} equivalent dropbacks"
            ),
            "margin_adjustment": "away QB expected points lost minus home QB expected points lost",
            "total_adjustment": "negative sum of home and away QB expected points lost",
            "lambda_selection": (
                "0.00-1.50 grid in 0.05 steps; each validation week chooses lambda "
                "using earlier OOF rows only; zero until 100 earlier OOF rows exist"
            ),
            "timing_limit": (
                "historical injury reports are a late-week/final-report proxy rather than "
                "timestamped early-week snapshots"
            ),
        },
        "margin": _target_report(
            games,
            features=GAME_MARGIN_FEATURES,
            target="home_margin",
            adjustment_kind="margin",
            min_train_rows=min_train_rows,
            holdout_season=holdout_season,
        ),
        "total": _target_report(
            games,
            features=GAME_TOTAL_FEATURES,
            target="total_points",
            adjustment_kind="total",
            min_train_rows=min_train_rows,
            holdout_season=holdout_season,
        ),
    }


def parse_args() -> argparse.Namespace:
    context = get_season_context()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=list(context.training_seasons))
    parser.add_argument("--holdout-season", type=int, default=context.prediction_season - 1)
    parser.add_argument("--min-train-rows", type=int, default=350)
    parser.add_argument("--lookback-weeks", type=int, default=4)
    parser.add_argument("--shrinkage-dropbacks", type=float, default=80.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qb_adjustment_benchmark.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_benchmark(
        args.seasons,
        min_train_rows=args.min_train_rows,
        holdout_season=args.holdout_season,
        lookback_weeks=args.lookback_weeks,
        shrinkage_dropbacks=args.shrinkage_dropbacks,
    )
    atomic_write_json(args.output, report)
    summary = {
        target: {
            split: {
                "base": report[target][split]["baseline"]["mae"],
                "adjusted": report[target][split]["adjusted"]["mae"],
                "change": report[target][split]["mae_change_vs_base"],
            }
            for split in ("development", "holdout", "all_oof")
        }
        for target in ("margin", "total")
    }
    print(json.dumps({"qb_loss": report["qb_loss_summary"], "results": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
