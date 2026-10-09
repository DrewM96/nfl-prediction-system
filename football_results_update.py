"""Refresh game outcomes without training models or rewriting frozen forecasts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cfb_prediction.client import CFBDClient
from cfb_prediction.config import CFB_DATA_DIR, CFB_PREDICTIONS_DIR
from cfb_prediction.data import normalize_games
from cfb_prediction.season import current_cfb_season
from nfl_prediction.config import PREDICTIONS_DIR, PROJECT_ROOT, get_season_context
from nfl_prediction.io import atomic_write_json
from nfl_prediction.results import performance_history, settle_schedule
from nfl_results_update import refresh_results


def refresh_cfb_results(
    schedules: pd.DataFrame, root: Path, output: Path, *, now: datetime, season: int
) -> dict[str, Any]:
    # An explicit upstream final is required; elapsed kickoff and partial scores
    # alone cannot settle a college game.
    schedules = schedules[schedules.season.eq(season)]
    writes = settle_schedule(root, schedules, sport="CFB")
    report = {"created_at": now.isoformat(), "season": season, "settlement_documents_added": writes}
    if writes:
        atomic_write_json(output, report)
        atomic_write_json(output.parent / "performance_history.json", performance_history(root))
    return report


def load_cfb_results(client: CFBDClient, season: int) -> pd.DataFrame:
    # Fetch only the scores endpoint, never historical features or models.
    records = client.get(
        "/games",
        params={"year": season, "seasonType": "regular", "classification": "fbs"},
        refresh=True,
    )
    return normalize_games(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=["nfl", "cfb"], required=True)
    parser.add_argument("--season", type=int)
    args = parser.parse_args()
    now = datetime.now(UTC)
    if args.sport == "nfl":
        import nflreadpy as nfl

        season = args.season or get_season_context(now).prediction_season
        schedules = nfl.load_schedules([season]).to_pandas()
        schedules = schedules[schedules.season.eq(season)]
        report = refresh_results(
            schedules,
            PREDICTIONS_DIR,
            PROJECT_ROOT / "data/nfl_results_summary.json",
            now=now,
            season=season,
            write_if_unchanged=False,
        )
        if report["settlement_documents_added"]:
            atomic_write_json(
                PROJECT_ROOT / "performance_history.json", performance_history(PREDICTIONS_DIR)
            )
    else:
        season = args.season or current_cfb_season(now)
        report = refresh_cfb_results(
            load_cfb_results(CFBDClient.from_environment(), season),
            CFB_PREDICTIONS_DIR,
            CFB_DATA_DIR / "results_summary.json",
            now=now,
            season=season,
        )
    print(
        f"{args.sport.upper()}: added {report['settlement_documents_added']} settlement documents"
    )


if __name__ == "__main__":
    main()
