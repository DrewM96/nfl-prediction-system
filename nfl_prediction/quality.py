"""Release gates and explicit provenance for NFL forecast inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_PBP_COLUMNS = {
    "game_id",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "yards_gained",
    "qb_dropback",
    "qb_hit",
    "sack",
    "interception",
    "fumble_lost",
}


def validate_source_schema(pbp: pd.DataFrame, schedules: pd.DataFrame) -> None:
    missing = REQUIRED_PBP_COLUMNS - set(pbp.columns)
    if missing:
        raise ValueError(f"Required PBP columns missing: {sorted(missing)}")
    if schedules["game_id"].duplicated().any():
        raise ValueError("Duplicate schedule game IDs; refusing ambiguous training rows")
    if {"game_id", "play_id"}.issubset(pbp) and pbp.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate PBP game/play IDs; refusing double-counted inputs")


def forecast_quality(game: pd.Series, roster_available: bool) -> dict[str, Any]:
    audit = game.get("input_audit", {})
    failures = []
    warnings = []
    required_stats = {
        "yards",
        "off_epa",
        "def_epa",
        "turnovers",
        "pressure_allowed",
        "pressure_generated",
    }
    for side in ("home", "away"):
        state = audit.get(side, {})
        if not state.get("sources_l8"):
            failures.append(f"{side}: no prior games")
        if state.get("missing_pbp_games_l4"):
            failures.append(f"{side}: missing recent PBP")
        missing = required_stats & set(state.get("imputed_fields_l4", []))
        if missing:
            failures.append(f"{side}: incomplete recent statistics {sorted(missing)}")
        missing_epa = sum(
            int(row.get("missing_epa_plays") or 0) for row in state.get("sources_l8", [])[-4:]
        )
        if missing_epa:
            warnings.append(
                f"{side}: {missing_epa} scrimmage EPA values are missing and use the legacy zero fallback"
            )
        if any(
            row.get("pbp_available")
            and (not row.get("offense_epa_plays") or not row.get("defense_epa_plays"))
            for row in state.get("sources_l8", [])[-4:]
        ):
            failures.append(f"{side}: no valid offensive or defensive EPA in a recent game")
    if not roster_available:
        warnings.append("Roster transition feed unavailable; continuity features are neutral")
    elif float(game.get("week", 1)) <= 4:
        for side, available in audit.get("roster_transition_available", {}).items():
            if not available:
                warnings.append(
                    f"{side}: roster transition join unavailable; continuity is neutral"
                )
    return {
        "status": "blocked" if failures else "degraded" if warnings else "complete",
        "failures": failures,
        "warnings": warnings,
        "teams": audit,
    }
