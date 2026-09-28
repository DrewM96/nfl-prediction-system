"""Predeclared season-transition candidates; promotion is never automatic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_prediction.features import (
    GAME_MARGIN_FEATURES,
    GAME_TOTAL_FEATURES,
    build_point_in_time_game_features,
)
from nfl_prediction.historical_market import prequential_component_blend
from nfl_prediction.io import atomic_write_json, sha256_file
from nfl_prediction.modeling import GAME_RIDGE_ALPHA, chronological_oof_predictions
from nfl_prediction.strength import StrengthConfig, strength_forecasts
from nfl_prediction.validation import paired_week_block_interval

CONFIGS = {"carry_50": StrengthConfig(carry=0.5), "carry_75": StrengthConfig(carry=0.75)}


def evaluate(frame: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    clean = frame.dropna(subset=[*features, target]).copy()
    actual, first, second, indices = chronological_oof_predictions(
        clean, features, target, min_train_rows=350, ridge_alpha=GAME_RIDGE_ALPHA
    )
    validation = clean.loc[indices].copy()
    predicted, _ = prequential_component_blend(validation, actual, first, second)
    return validation[["game_id", "season", "week", target]].assign(predicted=predicted)


def compare(rows: pd.DataFrame, target: str) -> dict:
    actual = rows[target].to_numpy()
    candidate, reference = rows.predicted.to_numpy(), rows.reference.to_numpy()
    interval = paired_week_block_interval(rows, actual, candidate, reference)
    return {
        "games": len(rows),
        "mae": float(np.abs(candidate - actual).mean()),
        "reference_mae": float(np.abs(reference - actual).mean()),
        "difference_95_interval": interval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/research"))
    parser.add_argument("--output", type=Path, default=Path("reports/strength-transition"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    schedules = pd.read_parquet(args.cache / "schedules.parquet")
    schedules = schedules[schedules.season.between(2020, 2025)]

    def load(prefix):
        return pd.concat(
            [
                pd.read_parquet(args.cache / f"{prefix}-{year}.parquet")
                for year in range(2020, 2026)
            ],
            ignore_index=True,
        )

    # Save the candidate contract before evaluating. 2025 is reused validation;
    # future frozen forecasts are needed for genuinely untouched confirmation.
    atomic_write_json(
        args.output / "experiment.json",
        {
            "started_at": datetime.now(UTC).isoformat(),
            "source_hashes": {
                str(path): sha256_file(path)
                for path in sorted(args.cache.glob("*.parquet"))
                if "2026" not in path.name
            },
            "code_hashes": {
                str(path): sha256_file(path)
                for path in [Path(__file__), *Path("nfl_prediction").glob("*.py")]
            },
            "configs": {name: asdict(config) for name, config in CONFIGS.items()},
            "selection": "For each outer season 2023-2025, choose using earlier OOF seasons only",
            "features": ["strength_margin", "strength_total", "strength_uncertainty"],
            "promotion": "No automatic promotion; report paired week-block intervals and early-season errors",
            "timing_limit": "Both core features and strength state are fixed before the week. Historical feeds are latest revised vintages, not archived publication-time feeds.",
        },
    )
    frame = build_point_in_time_game_features(
        schedules, load("pbp"), rosters=load("rosters"), snap_counts=load("snaps"), freeze_week=True
    ).games
    report = {}
    for target, features in (
        ("home_margin", GAME_MARGIN_FEATURES),
        ("total_points", GAME_TOTAL_FEATURES),
    ):
        print(f"Baseline {target}", flush=True)
        baseline = evaluate(frame, features, target).rename(columns={"predicted": "reference"})
        candidates = {}
        for name, config in CONFIGS.items():
            print(f"Candidate {target} {name}", flush=True)
            enriched = frame.merge(
                strength_forecasts(schedules, config),
                on=["game_id", "season", "week"],
                validate="one_to_one",
            )
            candidate = evaluate(
                enriched,
                features + ["strength_margin", "strength_total", "strength_uncertainty"],
                target,
            )
            candidate = candidate.merge(
                baseline[["game_id", "reference"]], on="game_id", validate="one_to_one"
            )
            candidate.to_csv(args.output / f"{target}-{name}.csv", index=False)
            candidates[name] = candidate
        selected = []
        choices = {}
        for season in (2023, 2024, 2025):
            scores = {
                "baseline": float(
                    np.abs(
                        baseline.loc[baseline.season.lt(season), "reference"]
                        - baseline.loc[baseline.season.lt(season), target]
                    ).mean()
                )
            }
            scores.update(
                {
                    name: float(
                        np.abs(
                            rows.loc[rows.season.lt(season), "predicted"]
                            - rows.loc[rows.season.lt(season), target]
                        ).mean()
                    )
                    for name, rows in candidates.items()
                }
            )
            winner = min(scores, key=scores.get)
            choices[str(season)] = {"chosen": winner, "prior_season_mae": scores}
            chosen = (
                baseline.assign(predicted=baseline.reference)
                if winner == "baseline"
                else candidates[winner]
            )
            selected.append(chosen[chosen.season.eq(season)])
        outer = pd.concat(selected, ignore_index=True)
        outer.to_csv(args.output / f"{target}-selected.csv", index=False)
        report[target] = {
            "by_season": {
                str(season): compare(outer[outer.season.eq(season)], target)
                for season in (2023, 2024, 2025)
            },
            "choices": choices,
            "outer": compare(outer, target),
            "weeks_1_4": compare(outer[outer.week.le(4)], target),
            "candidates": {
                name: {
                    "all": compare(rows[rows.season.ge(2023)], target),
                    "early": compare(rows[rows.season.ge(2023) & rows.week.le(4)], target),
                }
                for name, rows in candidates.items()
            },
        }
        atomic_write_json(args.output / "benchmark.json", report)
    print(args.output / "benchmark.json", flush=True)


if __name__ == "__main__":
    main()
