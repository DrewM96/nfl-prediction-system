"""Refresh outcomes independently of model training; never rewrite forecasts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_prediction.config import PREDICTIONS_DIR, PROJECT_ROOT, get_season_context
from nfl_prediction.io import atomic_write_json
from nfl_prediction.odds import _game_kickoff
from nfl_prediction.prop_results import refresh_player_results
from nfl_prediction.results import (
    forecast_rows,
    performance_history,
    select_forecasts,
    settle_schedule,
    summarize,
)


def refresh_results(
    schedules: pd.DataFrame,
    root: Path,
    output: Path,
    *,
    now: datetime,
    season: int | None = None,
    write_if_unchanged: bool = True,
) -> dict:
    # NFL schedules do not supply an explicit completed flag. Avoid settling
    # in-progress score fields; revisit them on the next daily run.
    completed = schedules.dropna(subset=["home_score", "away_score"]).copy()
    eligible = []
    for _, game in completed.iterrows():
        kickoff = _game_kickoff(game.to_dict())
        eligible.append(kickoff is not None and (now - kickoff).total_seconds() >= 8 * 3600)
    completed = completed.loc[eligible]
    writes = settle_schedule(root, completed)
    if not writes and not write_if_unchanged:
        return {"settlement_documents_added": 0, "season": season}
    rows = forecast_rows(root, as_of=now)
    if season is not None and not rows.empty:
        rows = rows[rows.season.eq(season)]
    reports = {}
    for name, policy, minutes in (
        ("first", "first", 60),
        ("latest_24h", "horizon", 1440),
        ("latest_60m", "horizon", 60),
    ):
        selected = select_forecasts(rows, policy=policy, horizon_minutes=minutes)
        reports[name] = {
            "margin": summarize(selected),
            "total": summarize(selected, target="total"),
            "selection": name,
            "note": "Latest available before horizon; not necessarily issued at horizon",
        }
    report = {
        "created_at": now.isoformat(),
        "season": season,
        "settlement_documents_added": writes,
        "policies": reports,
    }
    atomic_write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=get_season_context().prediction_season)
    parser.add_argument("--schedule", type=Path, help="Optional cached schedule parquet")
    args = parser.parse_args()
    if args.schedule:
        schedules = pd.read_parquet(args.schedule)
    else:
        import nflreadpy as nfl

        schedules = nfl.load_schedules([args.season]).to_pandas()
    schedules = schedules[schedules.season.eq(args.season)]
    report = refresh_results(
        schedules,
        PREDICTIONS_DIR,
        PROJECT_ROOT / "data/nfl_results_summary.json",
        now=datetime.now(UTC),
        season=args.season,
    )
    prop_writes = refresh_player_results(
        PREDICTIONS_DIR, schedules, args.season, now=datetime.now(UTC)
    )
    atomic_write_json(
        PROJECT_ROOT / "performance_history.json", performance_history(PREDICTIONS_DIR)
    )
    print(f"Added {report['settlement_documents_added']} settlement documents")
    print(f"Added {prop_writes} player settlement documents")


if __name__ == "__main__":
    main()
