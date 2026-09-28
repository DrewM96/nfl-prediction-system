"""Reconcile frozen NFL feature vectors against cached source records."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_prediction.features import GAME_FEATURES, build_point_in_time_game_features
from nfl_prediction.io import atomic_write_json, read_json, sha256_file
from nfl_prediction.quality import forecast_quality, validate_source_schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/research"))
    parser.add_argument("--release", type=Path, default=Path("data/nfl_release.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/input-reconciliation.json"))
    args = parser.parse_args()
    release = read_json(args.release)
    state = release["state"]
    seasons = state["update"]["training_seasons"]
    cutoff = pd.Timestamp(state["update"]["data_cutoff"])
    schedules = pd.read_parquet(args.cache / "schedules.parquet")
    schedules = schedules[schedules.season.isin(seasons)].copy()
    schedules.loc[pd.to_datetime(schedules.gameday).gt(cutoff), ["home_score", "away_score"]] = (
        np.nan
    )
    files = [args.cache / "schedules.parquet"]

    def load(prefix):
        paths = [args.cache / f"{prefix}-{year}.parquet" for year in seasons]
        files.extend(paths)
        return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

    pbp = load("pbp")
    pbp = pbp[pd.to_datetime(pbp.game_date).le(cutoff)]
    validate_source_schema(pbp, schedules)
    result = build_point_in_time_game_features(
        schedules, pbp, include_unplayed=True, rosters=load("rosters"), snap_counts=load("snaps")
    )
    indexed = result.games.set_index("game_id")
    rows = []
    for frozen in state["schedule"]:
        rebuilt = indexed.loc[frozen["game_id"]]
        differences = {
            key: {"frozen": frozen["features"][key], "reconstructed": float(rebuilt[key])}
            for key in GAME_FEATURES
            if not np.isclose(float(rebuilt[key]), frozen["features"][key], atol=1e-8, rtol=0)
        }
        rows.append(
            {
                "game_id": frozen["game_id"],
                "differences": differences,
                "quality": forecast_quality(rebuilt, True),
            }
        )
    atomic_write_json(
        args.output,
        {
            "release_hash": release["model_hash"],
            "cutoff": str(cutoff.date()),
            "source_files": {str(path): sha256_file(path) for path in files},
            "limitation": "Latest revised source vintage truncated to release game-date cutoff; not original download bytes",
            "games": rows,
        },
    )
    print(
        f"{len(rows)} games; {sum(bool(row['differences']) for row in rows)} with feature differences; "
        f"{sum(row['quality']['status'] == 'blocked' for row in rows)} blocked by source coverage"
    )


if __name__ == "__main__":
    main()
