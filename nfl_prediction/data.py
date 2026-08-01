from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _to_pandas(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


@dataclass
class NFLData:
    pbp: pd.DataFrame
    schedules: pd.DataFrame
    rosters: pd.DataFrame
    injuries: pd.DataFrame
    snap_counts: pd.DataFrame


def _load_by_available_season(loader: Any, seasons: list[int]) -> pd.DataFrame:
    """Load every published season while skipping not-yet-created feeds."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            frame = _to_pandas(loader(season))
        except (FileNotFoundError, ValueError):
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_nflverse_data(seasons: list[int]) -> NFLData:
    """Load maintained nflverse data through nflreadpy.

    Optional feeds fail independently; play-by-play and schedules are required.
    """
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise RuntimeError(
            "nflreadpy is required. Install the locked project dependencies before updating."
        ) from exc

    pbp = _load_by_available_season(nfl.load_pbp, seasons)
    if pbp.empty:
        raise RuntimeError(f"No play-by-play data is published for requested seasons {seasons}.")

    schedules = _to_pandas(nfl.load_schedules(seasons))
    schedules = schedules[schedules["season"].isin(seasons)].copy()
    if schedules.empty:
        raise RuntimeError(f"No schedule data is published for requested seasons {seasons}.")

    def optional(loader_name: str) -> pd.DataFrame:
        loader = getattr(nfl, loader_name, None)
        if loader is None:
            return pd.DataFrame()
        try:
            frame = _load_by_available_season(loader, seasons)
        except Exception as exc:
            LOGGER.warning("Optional nflverse feed %s failed: %s", loader_name, exc)
            return pd.DataFrame()
        if "season" in frame:
            frame = frame[frame["season"].isin(seasons)]
        return frame

    rosters = optional("load_rosters_weekly")
    roster_seasons = (
        set(pd.to_numeric(rosters["season"], errors="coerce").dropna())
        if "season" in rosters
        else set()
    )
    missing_roster_seasons = [season for season in seasons if season not in roster_seasons]
    seasonal_loader = getattr(nfl, "load_rosters", None)
    if missing_roster_seasons and seasonal_loader is not None:
        try:
            seasonal_rosters = _load_by_available_season(seasonal_loader, missing_roster_seasons)
        except Exception as exc:
            LOGGER.warning("Offseason nflverse roster fallback failed: %s", exc)
        else:
            rosters = pd.concat([rosters, seasonal_rosters], ignore_index=True)

    return NFLData(
        pbp=pbp,
        schedules=schedules,
        rosters=rosters,
        injuries=optional("load_injuries"),
        snap_counts=optional("load_snap_counts"),
    )
