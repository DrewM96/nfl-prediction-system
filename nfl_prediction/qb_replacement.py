from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .injuries import injury_unavailability_weight

QB_REPLACEMENT_TEAM_FEATURES = [
    "qb_starter_reported",
    "qb_unavailability_weight",
    "qb_starter_epa_per_dropback",
    "qb_backup_epa_per_dropback",
    "qb_value_gap_epa_per_dropback",
    "qb_expected_dropbacks",
    "qb_expected_points_lost",
]

QB_REPLACEMENT_GAME_FEATURES = [
    f"{side}_{feature}"
    for side in ("home", "away")
    for feature in QB_REPLACEMENT_TEAM_FEATURES
]

QB_REPLACEMENT_CANDIDATE_GROUPS = {
    "qb_availability": [
        "home_qb_starter_reported",
        "home_qb_unavailability_weight",
        "away_qb_starter_reported",
        "away_qb_unavailability_weight",
    ],
    "qb_value_gap": [
        "home_qb_value_gap_epa_per_dropback",
        "away_qb_value_gap_epa_per_dropback",
    ],
    "qb_expected_points_loss": [
        "home_qb_expected_points_lost",
        "away_qb_expected_points_lost",
    ],
}


def _regular(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "season_type" in result:
        result = result[result["season_type"].eq("REG")]
    elif "game_type" in result:
        result = result[result["game_type"].eq("REG")]
    return result


def _prepare_qb_dropbacks(pbp: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "week", "posteam", "passer_player_id", "epa"}
    if pbp.empty or not required.issubset(pbp):
        return pd.DataFrame(
            columns=["season", "week", "team", "player_id", "epa", "success", "dropbacks"]
        )

    frame = _regular(pbp)
    frame = frame.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame["epa"] = pd.to_numeric(frame["epa"], errors="coerce")
    qb_dropback = pd.to_numeric(
        frame.get("qb_dropback", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    sack = pd.to_numeric(
        frame.get("sack", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    pass_attempt = pd.to_numeric(
        frame.get("pass_attempt", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    frame = frame[(qb_dropback.eq(1) | sack.eq(1) | pass_attempt.eq(1))].copy()
    frame = frame[
        frame["season"].notna()
        & frame["week"].notna()
        & frame["posteam"].notna()
        & frame["passer_player_id"].notna()
    ]
    if frame.empty:
        return pd.DataFrame(
            columns=["season", "week", "team", "player_id", "epa", "success", "dropbacks"]
        )

    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["team"] = frame["posteam"].astype(str)
    frame["player_id"] = frame["passer_player_id"].astype(str)
    frame["success"] = (
        pd.to_numeric(
            frame.get("success", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        .fillna(frame["epa"].gt(0).astype(float))
        .astype(float)
    )
    frame["dropbacks"] = 1.0
    return frame[["season", "week", "team", "player_id", "epa", "success", "dropbacks"]]


def _qb_history(
    dropbacks: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    lookback_weeks: int,
) -> pd.DataFrame:
    current = dropbacks[
        dropbacks["season"].eq(season)
        & dropbacks["team"].eq(team)
        & dropbacks["week"].lt(week)
    ].copy()
    if not current.empty:
        weeks = sorted(current["week"].unique())[-lookback_weeks:]
        return current[current["week"].isin(weeks)]

    prior = dropbacks[
        dropbacks["season"].eq(season - 1) & dropbacks["team"].eq(team)
    ].copy()
    if prior.empty:
        return prior
    weeks = sorted(prior["week"].unique())[-lookback_weeks:]
    return prior[prior["week"].isin(weeks)]


def _league_epa_prior(
    dropbacks: pd.DataFrame,
    *,
    season: int,
    week: int,
) -> float:
    prior = dropbacks[
        (dropbacks["season"] < season)
        | ((dropbacks["season"].eq(season)) & dropbacks["week"].lt(week))
    ]
    if prior.empty:
        return 0.0
    recent = prior[prior["season"].ge(season - 1)]
    if not recent.empty:
        prior = recent
    value = float(prior["epa"].mean())
    return value if np.isfinite(value) else 0.0


def _shrunk_epa_per_dropback(
    history: pd.DataFrame,
    player_id: str,
    *,
    league_prior: float,
    prior_dropbacks: float,
) -> tuple[float, float]:
    player = history[history["player_id"].eq(player_id)]
    n = float(player["dropbacks"].sum())
    epa_sum = float(player["epa"].fillna(0.0).sum())
    denominator = n + prior_dropbacks
    value = (
        (epa_sum + league_prior * prior_dropbacks) / denominator
        if denominator > 0
        else league_prior
    )
    return float(value), n


def _starter_and_backup(
    history: pd.DataFrame,
    *,
    league_prior: float,
    prior_dropbacks: float,
) -> tuple[str | None, str | None, float, float, float]:
    if history.empty:
        return None, None, league_prior, league_prior, 35.0

    usage = (
        history.groupby("player_id", as_index=False)["dropbacks"]
        .sum()
        .sort_values(["dropbacks", "player_id"], ascending=[False, True])
    )
    starter = str(usage.iloc[0]["player_id"]) if not usage.empty else None
    backup = str(usage.iloc[1]["player_id"]) if len(usage) > 1 else None
    starter_value = league_prior
    backup_value = league_prior

    if starter:
        starter_value, _ = _shrunk_epa_per_dropback(
            history,
            starter,
            league_prior=league_prior,
            prior_dropbacks=prior_dropbacks,
        )
    if backup:
        backup_value, _ = _shrunk_epa_per_dropback(
            history,
            backup,
            league_prior=league_prior,
            prior_dropbacks=prior_dropbacks,
        )

    team_week_dropbacks = history.groupby("week")["dropbacks"].sum()
    expected_dropbacks = float(team_week_dropbacks.mean()) if not team_week_dropbacks.empty else 35.0
    return starter, backup, starter_value, backup_value, expected_dropbacks


def build_qb_replacement_table(
    injuries: pd.DataFrame | None,
    pbp: pd.DataFrame | None,
    *,
    lookback_weeks: int = 4,
    shrinkage_dropbacks: float = 80.0,
) -> pd.DataFrame:
    """Build research-only team-week QB replacement-value features.

    Starter identity is inferred strictly from prior team dropback usage. QB value
    is a shrunk prior-game EPA/dropback estimate. Expected points lost scales the
    starter-vs-backup EPA gap by expected team dropbacks and injury severity.

    Historical nflverse injury rows are a late-week/final-report proxy, not a
    timestamped reconstruction of what an earlier forecast knew.
    """
    injuries = injuries if injuries is not None else pd.DataFrame()
    pbp = pbp if pbp is not None else pd.DataFrame()
    required = {"season", "week", "team", "position", "gsis_id"}
    if injuries.empty or not required.issubset(injuries):
        return pd.DataFrame(columns=["season", "week", "team", *QB_REPLACEMENT_TEAM_FEATURES])

    dropbacks = _prepare_qb_dropbacks(pbp)
    if dropbacks.empty:
        return pd.DataFrame(columns=["season", "week", "team", *QB_REPLACEMENT_TEAM_FEATURES])

    reports = injuries.copy()
    reports["season"] = pd.to_numeric(reports["season"], errors="coerce")
    reports["week"] = pd.to_numeric(reports["week"], errors="coerce")
    reports = reports[reports["season"].notna() & reports["week"].notna()].copy()
    reports["season"] = reports["season"].astype(int)
    reports["week"] = reports["week"].astype(int)
    reports["team"] = reports["team"].fillna("").astype(str)
    reports["position"] = reports["position"].fillna("").astype(str).str.upper()
    reports["gsis_id"] = reports["gsis_id"].fillna("").astype(str)
    reports = reports[reports["position"].eq("QB")].copy()

    covered = injuries[["season", "week", "team"]].copy()
    covered["season"] = pd.to_numeric(covered["season"], errors="coerce")
    covered["week"] = pd.to_numeric(covered["week"], errors="coerce")
    covered = covered.dropna().drop_duplicates()
    covered["season"] = covered["season"].astype(int)
    covered["week"] = covered["week"].astype(int)
    covered["team"] = covered["team"].astype(str)

    report_groups = {
        (int(season), int(week), str(team)): group
        for (season, week, team), group in reports.groupby(
            ["season", "week", "team"], sort=False
        )
    }

    rows: list[dict[str, Any]] = []
    for season, week, team in covered.itertuples(index=False, name=None):
        history = _qb_history(
            dropbacks,
            season=int(season),
            week=int(week),
            team=str(team),
            lookback_weeks=lookback_weeks,
        )
        league_prior = _league_epa_prior(
            dropbacks,
            season=int(season),
            week=int(week),
        )
        starter, backup, starter_value, backup_value, expected_dropbacks = _starter_and_backup(
            history,
            league_prior=league_prior,
            prior_dropbacks=shrinkage_dropbacks,
        )

        team_reports = report_groups.get((int(season), int(week), str(team)))
        starter_reported = 0.0
        severity = 0.0
        if starter and team_reports is not None:
            matching = team_reports[team_reports["gsis_id"].eq(starter)]
            if not matching.empty:
                starter_reported = 1.0
                severity = max(
                    injury_unavailability_weight(row)
                    for _, row in matching.iterrows()
                )

        value_gap = float(starter_value - backup_value)
        expected_points_lost = float(severity * value_gap * expected_dropbacks)
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team": str(team),
                "qb_starter_reported": starter_reported,
                "qb_unavailability_weight": float(severity),
                "qb_starter_epa_per_dropback": float(starter_value),
                "qb_backup_epa_per_dropback": float(backup_value),
                "qb_value_gap_epa_per_dropback": value_gap,
                "qb_expected_dropbacks": expected_dropbacks,
                "qb_expected_points_lost": expected_points_lost,
                "qb_starter_id": starter,
                "qb_backup_id": backup,
            }
        )

    return pd.DataFrame(rows).sort_values(["season", "week", "team"]).reset_index(drop=True)


def attach_qb_replacement_features(
    games: pd.DataFrame,
    qb_table: pd.DataFrame,
) -> pd.DataFrame:
    result = games.copy()
    for side in ("home", "away"):
        for feature in QB_REPLACEMENT_TEAM_FEATURES:
            result[f"{side}_{feature}"] = 0.0
    if result.empty or qb_table.empty:
        return result

    indexed = qb_table.set_index(["season", "week", "team"])
    for index, game in result.iterrows():
        season = int(game["season"])
        week = int(game["week"])
        for side in ("home", "away"):
            key = (season, week, str(game[f"{side}_team"]))
            if key not in indexed.index:
                continue
            row = indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            for feature in QB_REPLACEMENT_TEAM_FEATURES:
                result.at[index, f"{side}_{feature}"] = float(row[feature])
    return result
