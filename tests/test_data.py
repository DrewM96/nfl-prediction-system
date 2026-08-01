from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from nfl_prediction.data import load_nflverse_data


def test_future_schedule_does_not_require_future_pbp(monkeypatch) -> None:
    def by_season(season: int) -> pd.DataFrame:
        if season > 2025:
            raise ValueError("not published")
        return pd.DataFrame({"season": [season], "game_id": [str(season)]})

    fake = SimpleNamespace(
        load_pbp=by_season,
        load_schedules=lambda seasons: pd.DataFrame(
            {"season": seasons, "game_id": [f"schedule-{season}" for season in seasons]}
        ),
        load_rosters_weekly=by_season,
        load_rosters=lambda season: pd.DataFrame(
            {"season": [season], "game_id": [f"offseason-roster-{season}"]}
        ),
        load_injuries=by_season,
        load_snap_counts=by_season,
    )
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)

    result = load_nflverse_data([2024, 2025, 2026])

    assert set(result.pbp["season"]) == {2024, 2025}
    assert set(result.schedules["season"]) == {2024, 2025, 2026}
    assert set(result.injuries["season"]) == {2024, 2025}
    assert set(result.rosters["season"]) == {2024, 2025, 2026}
