from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_prediction.io import atomic_write_json

from .client import CFBDClient
from .config import CFB_FOUNDATION_PATH
from .data import CFBFoundationData, normalize_calendar, normalize_games, normalize_teams


def load_foundation_data(
    client: CFBDClient,
    season: int,
    *,
    refresh: bool = False,
) -> CFBFoundationData:
    return CFBFoundationData(
        teams=normalize_teams(client.get("/teams/fbs", params={"year": season}, refresh=refresh)),
        games=normalize_games(
            client.get(
                "/games",
                params={
                    "year": season,
                    "seasonType": "regular",
                    "classification": "fbs",
                },
                refresh=refresh,
            )
        ),
        calendar=normalize_calendar(
            client.get("/calendar", params={"year": season}, refresh=refresh)
        ),
    )


def build_foundation_summary(
    data: CFBFoundationData,
    season: int,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = generated_at or datetime.now(UTC)
    games = data.games
    calendar = data.calendar
    regular_calendar = calendar[calendar["season_type"].eq("regular")]
    completed = int(games["completed"].fillna(False).astype(bool).sum()) if not games.empty else 0
    start = regular_calendar["start_date"].min() if not regular_calendar.empty else None
    end = regular_calendar["end_date"].max() if not regular_calendar.empty else None
    return {
        "schema_version": 1,
        "sport": "college_football",
        "status": "data_ready",
        "prediction_season": int(season),
        "created_at": timestamp.isoformat(),
        "data_cutoff": timestamp.date().isoformat(),
        "source": "CollegeFootballData REST API v2",
        "raw_data_published": False,
        "team_count": int(len(data.teams)),
        "scheduled_game_count": int(len(games)),
        "fbs_vs_fbs_game_count": int(games["fbs_vs_fbs"].sum()) if not games.empty else 0,
        "completed_game_count": completed,
        "calendar_week_count": int(len(regular_calendar)),
        "season_start": None if pd.isna(start) else start.isoformat(),
        "season_end": None if pd.isna(end) else end.isoformat(),
        "models": {},
        "next_stage": "historical feature construction and chronological game-model benchmark",
    }


def run_foundation_update(
    season: int,
    *,
    refresh: bool = False,
    output_path: Path = CFB_FOUNDATION_PATH,
    client: CFBDClient | None = None,
) -> dict[str, Any]:
    active_client = client or CFBDClient.from_environment()
    summary = build_foundation_summary(
        load_foundation_data(active_client, season, refresh=refresh),
        season,
    )
    atomic_write_json(output_path, summary)
    return summary
