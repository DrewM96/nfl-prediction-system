"""Download public research inputs into the ignored local cache."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl

from nfl_prediction.config import get_season_context
from nfl_prediction.io import atomic_write_json, sha256_file

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "game_date",
    "season_type",
    "posteam",
    "defteam",
    "play_type",
    "yards_gained",
    "epa",
    "qb_dropback",
    "qb_hit",
    "sack",
    "interception",
    "fumble_lost",
    "down",
    "qtr",
    "score_differential",
    "success",
    "passer_player_id",
    "pass_attempt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(range(2020, get_season_context().prediction_season + 1)),
    )
    parser.add_argument("--cache", type=Path, default=Path("data/cache/research"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    def save(frame, path):
        frame.write_parquet(path)
        atomic_write_json(
            path.with_suffix(".metadata.json"),
            {
                "acquired_at": datetime.now(UTC).isoformat(),
                "source": "nflverse via nflreadpy",
                "rows": frame.height,
                "sha256": sha256_file(path),
            },
        )
        print(path, frame.shape, flush=True)

    save(nfl.load_schedules(args.seasons), args.cache / "schedules.parquet")
    for season in args.seasons:
        for label, loader in (
            ("pbp", nfl.load_pbp),
            ("rosters", nfl.load_rosters_weekly),
            ("snaps", nfl.load_snap_counts),
            ("injuries", nfl.load_injuries),
        ):
            path = args.cache / f"{label}-{season}.parquet"
            if (
                path.exists()
                and not args.refresh
                and season < get_season_context().prediction_season
            ):
                continue
            frame = loader(season)
            if label == "pbp":
                frame = frame.select([column for column in PBP_COLUMNS if column in frame.columns])
            save(frame, path)


if __name__ == "__main__":
    main()
