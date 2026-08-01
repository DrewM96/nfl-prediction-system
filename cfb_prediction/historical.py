from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from .client import CFBDApiError, CFBDClient
from .data import (
    CFBHistoricalData,
    normalize_advanced_game_stats,
    normalize_games,
    normalize_lines,
    normalize_portal,
    normalize_recruiting,
    normalize_returning_production,
    normalize_talent,
)

LOGGER = logging.getLogger(__name__)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _optional_season(
    client: CFBDClient,
    endpoint: str,
    params: dict[str, Any],
    normalizer: Callable[[Any], pd.DataFrame],
    *,
    refresh: bool,
) -> pd.DataFrame:
    try:
        return normalizer(client.get(endpoint, params=params, refresh=refresh))
    except CFBDApiError as exc:
        LOGGER.warning("Optional CFBD endpoint %s failed for %s: %s", endpoint, params, exc)
        return pd.DataFrame()


def load_historical_data(
    client: CFBDClient,
    seasons: list[int],
    *,
    refresh: bool = False,
) -> CFBHistoricalData:
    """Load cached season-level data with games and advanced stats required."""
    games: list[pd.DataFrame] = []
    advanced: list[pd.DataFrame] = []
    returning: list[pd.DataFrame] = []
    portal: list[pd.DataFrame] = []
    talent: list[pd.DataFrame] = []
    recruiting: list[pd.DataFrame] = []
    lines: list[pd.DataFrame] = []

    for season in seasons:
        common = {"year": season, "seasonType": "regular"}
        games.append(
            normalize_games(
                client.get("/games", params={**common, "classification": "fbs"}, refresh=refresh)
            )
        )
        advanced.append(
            normalize_advanced_game_stats(
                client.get("/stats/game/advanced", params=common, refresh=refresh)
            )
        )
        returning.append(
            _optional_season(
                client,
                "/player/returning",
                {"year": season},
                normalize_returning_production,
                refresh=refresh,
            )
        )
        portal.append(
            _optional_season(
                client,
                "/player/portal",
                {"year": season},
                normalize_portal,
                refresh=refresh,
            )
        )
        talent.append(
            _optional_season(
                client,
                "/talent",
                {"year": season},
                normalize_talent,
                refresh=refresh,
            )
        )
        recruiting.append(
            _optional_season(
                client,
                "/recruiting/teams",
                {"year": season},
                normalize_recruiting,
                refresh=refresh,
            )
        )
        lines.append(
            _optional_season(
                client,
                "/lines",
                common,
                normalize_lines,
                refresh=refresh,
            )
        )

    return CFBHistoricalData(
        games=_concat(games),
        advanced=_concat(advanced),
        returning=_concat(returning),
        portal=_concat(portal),
        talent=_concat(talent),
        recruiting=_concat(recruiting),
        lines=_concat(lines),
    )
