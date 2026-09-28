"""Record a separate prospective QB comparison against an existing frozen release."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nfl_prediction.io import atomic_write_json, read_json
from nfl_prediction.lineup import attach_lineup_shadow, lineup_table
from nfl_prediction.odds import _game_kickoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/research"))
    args = parser.parse_args()
    release = read_json("data/nfl_release.json")
    now = datetime.now(UTC)
    games = [
        game
        for game in release["state"]["schedule"]
        if _game_kickoff(game) and _game_kickoff(game) > now
    ]
    if not games:
        print("No unstarted games in the current frozen release")
        return
    season, week = games[0]["season"], games[0]["week"]
    injury_path = args.cache / f"injuries-{season}.parquet"
    injuries = pd.read_parquet(injury_path)
    rosters = pd.read_parquet(args.cache / f"rosters-{season}.parquet")
    pbp = pd.concat(
        [pd.read_parquet(args.cache / f"pbp-{year}.parquet") for year in (season - 1, season)],
        ignore_index=True,
    )
    captured = datetime.fromtimestamp(injury_path.stat().st_mtime, tz=UTC)
    fresh = int(injuries.week.max()) == week and (now - captured).total_seconds() <= 48 * 3600
    table = lineup_table(injuries, pbp, rosters, season=season, week=week)
    predictions = attach_lineup_shadow(
        games, table, captured_at=captured.isoformat(), as_of=now, fresh=fresh
    )
    path = Path("data/lineup_research") / f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.json"
    atomic_write_json(
        path,
        {
            "created_at": now.isoformat(),
            "base_run": release["state"]["update"]["ledger_run"],
            "base_model_hash": release["model_hash"],
            "predictions": predictions,
            "research_only": True,
            "note": "Cache acquisition time; official publication timestamp unavailable",
        },
    )
    print(path)


if __name__ == "__main__":
    main()
