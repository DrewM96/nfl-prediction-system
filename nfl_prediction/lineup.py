"""Research-only QB changes relative to participation already in team form."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .odds import parse_timestamp
from .qb_replacement import (
    _player_prior_history,
    _prepare_qb_dropbacks,
    _shrunk_epa_per_dropback,
    build_qb_replacement_table,
)


def relative_qb_points(
    baseline_value: float,
    starter_value: float,
    backup_value: float,
    unavailable_probability: float,
    dropbacks: float,
) -> float:
    if not 0 <= unavailable_probability <= 1 or dropbacks < 0:
        raise ValueError("Invalid QB probability or workload")
    expected = (
        1 - unavailable_probability
    ) * starter_value + unavailable_probability * backup_value
    return (expected - baseline_value) * dropbacks


def reserve_reports(injuries: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Include explicit reserve QB absences, preserving historical team/week keys."""
    required = {"season", "week", "team", "gsis_id", "position", "status"}
    if not required.issubset(rosters):
        return injuries.copy()
    reserve = rosters[
        rosters.position.eq("QB") & rosters.status.isin({"IR", "PUP", "NFI", "SUS", "RES"})
    ].copy()
    reserve["report_status"] = "Out"
    return pd.concat(
        [injuries, reserve[["season", "week", "team", "gsis_id", "position", "report_status"]]],
        ignore_index=True,
    )


def lineup_table(
    injuries: pd.DataFrame,
    pbp: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    season: int | None = None,
    week: int | None = None,
) -> dict:
    if season is not None:
        if not {"season", "week"}.issubset(injuries) or week is None:
            return {}
        injuries = injuries[injuries.season.eq(season) & injuries.week.eq(week)]
        if {"season", "week"}.issubset(rosters):
            rosters = rosters[rosters.season.eq(season) & rosters.week.le(week)]
    table = build_qb_replacement_table(reserve_reports(injuries, rosters), pbp, rosters)
    dropbacks = _prepare_qb_dropbacks(pbp)
    output = {}
    for row in table.itertuples():
        if week is not None and row.week != week:
            continue
        key = (int(row.season), int(row.week), str(row.team))
        record = {
            "eligible": False,
            "starter_id": row.qb_starter_id,
            "backup_id": row.qb_backup_id,
            "starter_basis": "inferred prior usage; not a confirmed depth chart",
            "probability_basis": "fixed designation proxy; not calibrated",
            "reason": "missing_starter_or_backup",
        }
        output[key] = record
        if not row.qb_starter_id or not row.qb_backup_id:
            continue
        prior = dropbacks[
            (
                dropbacks.season.lt(row.season)
                | (dropbacks.season.eq(row.season) & dropbacks.week.lt(row.week))
            )
            & dropbacks.season.ge(row.season - 1)
        ]
        team = prior[prior.team.eq(row.team)]
        keys = team[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(4)
        team = team.merge(keys, on=["season", "week"])
        if team.empty:
            record["reason"] = "missing_baseline_participation"
            continue
        # Equal game weights match the core form window; within each game use
        # QB dropback share. EPA values are shrunk estimates known before the week.
        counts = team.groupby(["season", "week", "player_id"]).size().rename("plays").reset_index()
        counts["share"] = counts.plays / counts.groupby(["season", "week"]).plays.transform("sum")
        exposures = (counts.groupby("player_id").share.sum() / len(keys)).to_dict()
        league = float(prior.epa.mean()) if not prior.empty else 0.0
        baseline = 0.0
        for player, share in exposures.items():
            history = _player_prior_history(
                dropbacks, season=int(row.season), week=int(row.week), player_id=player
            )
            value, _ = _shrunk_epa_per_dropback(
                history, player, league_prior=league, prior_dropbacks=80
            )
            baseline += value * share
        change = relative_qb_points(
            baseline,
            row.qb_starter_epa_per_dropback,
            row.qb_backup_epa_per_dropback,
            row.qb_unavailability_weight,
            row.qb_expected_dropbacks,
        )
        record.update(
            eligible=True,
            reason=None,
            baseline_epa=baseline,
            baseline_qb_shares=exposures,
            expected_points_change=change,
            unavailability_probability=row.qb_unavailability_weight,
            source_weeks=keys.to_dict("records"),
        )
    return output


def attach_lineup_shadow(
    predictions: list[dict], table: dict, *, captured_at: str, as_of: datetime, fresh: bool
) -> list[dict]:
    if parse_timestamp(captured_at) > as_of:
        raise ValueError("Future injury snapshot")
    output = []
    for game in predictions:
        home = table.get((int(game["season"]), int(game["week"]), game["home_team"]), {})
        away = table.get((int(game["season"]), int(game["week"]), game["away_team"]), {})
        eligible = bool(fresh and home.get("eligible") and away.get("eligible"))
        delta = (
            home["expected_points_change"] - away["expected_points_change"] if eligible else None
        )
        base = (game.get("football_only") or {}).get("home_margin", game["predicted_home_margin"])
        output.append(
            {
                **game,
                "lineup_shadow": {
                    "version": "baseline-relative-qb-v1",
                    "research_only": True,
                    "applied_to_published_forecast": False,
                    "eligible": eligible,
                    "captured_at": captured_at,
                    "fresh_for_week": fresh,
                    "home": home,
                    "away": away,
                    "margin_change": delta,
                    "shadow_margin": base + delta if eligible else None,
                    "coefficient": 1.0,
                    "coefficient_validated": False,
                },
            }
        )
    return output
