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
    f"{side}_{feature}" for side in ("home", "away") for feature in QB_REPLACEMENT_TEAM_FEATURES
]

QB_SHADOW_LAMBDA = 1.5


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


def _qb_groups(
    dropbacks: pd.DataFrame,
) -> dict[tuple[int, str], pd.DataFrame]:
    if dropbacks.empty:
        return {}
    return {
        (int(season), str(team)): group.copy()
        for (season, team), group in dropbacks.groupby(["season", "team"], sort=False)
    }


def _qb_history(
    groups: dict[tuple[int, str], pd.DataFrame],
    *,
    season: int,
    week: int,
    team: str,
    lookback_weeks: int,
) -> pd.DataFrame:
    current = groups.get((season, team), pd.DataFrame()).copy()
    if not current.empty:
        current = current[current["week"].lt(week)]
    if not current.empty:
        weeks = sorted(current["week"].unique())[-lookback_weeks:]
        return current[current["week"].isin(weeks)]

    prior = groups.get((season - 1, team), pd.DataFrame()).copy()
    if prior.empty:
        return prior
    weeks = sorted(prior["week"].unique())[-lookback_weeks:]
    return prior[prior["week"].isin(weeks)]


def _league_epa_priors(
    dropbacks: pd.DataFrame,
    keys: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    priors: dict[tuple[int, int], float] = {}
    for season, week in sorted(set(keys)):
        prior = dropbacks[
            (dropbacks["season"] < season)
            | ((dropbacks["season"].eq(season)) & dropbacks["week"].lt(week))
        ]
        if not prior.empty:
            recent = prior[prior["season"].ge(season - 1)]
            if not recent.empty:
                prior = recent
        value = float(prior["epa"].mean()) if not prior.empty else 0.0
        priors[(season, week)] = value if np.isfinite(value) else 0.0
    return priors


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


def _roster_qb_groups(
    rosters: pd.DataFrame | None,
) -> dict[tuple[int, str], pd.DataFrame]:
    if rosters is None or rosters.empty:
        return {}
    required = {"season", "team", "position", "gsis_id"}
    if not required.issubset(rosters):
        return {}
    frame = rosters.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(
        frame.get("week", pd.Series(1, index=frame.index)), errors="coerce"
    ).fillna(1)
    frame = frame[frame["season"].notna() & frame["team"].notna() & frame["gsis_id"].notna()].copy()
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["team"] = frame["team"].astype(str)
    frame["position"] = frame["position"].fillna("").astype(str).str.upper()
    frame["gsis_id"] = frame["gsis_id"].astype(str)
    frame = frame[frame["position"].eq("QB")]
    return {
        (int(season), str(team)): group.copy()
        for (season, team), group in frame.groupby(["season", "team"], sort=False)
    }


def _current_roster_qbs(
    groups: dict[tuple[int, str], pd.DataFrame],
    *,
    season: int,
    week: int,
    team: str,
) -> list[str]:
    frame = groups.get((season, team))
    if frame is None or frame.empty:
        return []
    available = frame[frame["week"].le(week)]
    if available.empty:
        available = frame[frame["week"].eq(frame["week"].min())]
    else:
        available = available[available["week"].eq(available["week"].max())]
    return sorted(set(available["gsis_id"].astype(str)))


def _player_prior_history(
    dropbacks: pd.DataFrame,
    *,
    season: int,
    week: int,
    player_id: str,
    lookback_weeks: int = 8,
) -> pd.DataFrame:
    history = dropbacks[
        dropbacks["player_id"].eq(player_id)
        & (
            (dropbacks["season"] < season)
            | (dropbacks["season"].eq(season) & dropbacks["week"].lt(week))
        )
    ].copy()
    if history.empty:
        return history
    history = history[history["season"].ge(season - 1)]
    keys = (
        history[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .tail(lookback_weeks)
    )
    return history.merge(keys, on=["season", "week"], how="inner")


def _fallback_roster_order(
    dropbacks: pd.DataFrame,
    roster_qbs: list[str],
    *,
    season: int,
    week: int,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for player_id in roster_qbs:
        history = _player_prior_history(
            dropbacks,
            season=season,
            week=week,
            player_id=player_id,
        )
        scored.append((float(history["dropbacks"].sum()), player_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] <= 0:
        return []
    return [player_id for _, player_id in scored]


def _starter_and_backup(
    history: pd.DataFrame,
    *,
    dropbacks: pd.DataFrame,
    roster_qbs: list[str],
    season: int,
    week: int,
    league_prior: float,
    prior_dropbacks: float,
) -> tuple[str | None, str | None, float, float, float]:
    usage = (
        history.groupby("player_id", as_index=False)["dropbacks"]
        .sum()
        .sort_values(["dropbacks", "player_id"], ascending=[False, True])
        if not history.empty
        else pd.DataFrame(columns=["player_id", "dropbacks"])
    )
    ordered = [str(value) for value in usage["player_id"].tolist()]
    if roster_qbs:
        roster_set = set(roster_qbs)
        ordered = [player_id for player_id in ordered if player_id in roster_set]
    if not ordered:
        ordered = _fallback_roster_order(
            dropbacks,
            roster_qbs,
            season=season,
            week=week,
        )

    starter = ordered[0] if ordered else None
    backup = ordered[1] if len(ordered) > 1 else None
    if backup is None and roster_qbs:
        remaining = [player_id for player_id in roster_qbs if player_id != starter]
        fallback_order = _fallback_roster_order(
            dropbacks,
            remaining,
            season=season,
            week=week,
        )
        backup = fallback_order[0] if fallback_order else None

    starter_value = league_prior
    backup_value = league_prior
    if starter:
        starter_history = _player_prior_history(
            dropbacks,
            season=season,
            week=week,
            player_id=starter,
        )
        starter_value, _ = _shrunk_epa_per_dropback(
            starter_history,
            starter,
            league_prior=league_prior,
            prior_dropbacks=prior_dropbacks,
        )
    if backup:
        backup_history = _player_prior_history(
            dropbacks,
            season=season,
            week=week,
            player_id=backup,
        )
        backup_value, _ = _shrunk_epa_per_dropback(
            backup_history,
            backup,
            league_prior=league_prior,
            prior_dropbacks=prior_dropbacks,
        )

    team_week_dropbacks = (
        history.groupby("week")["dropbacks"].sum()
        if not history.empty and "week" in history
        else pd.Series(dtype=float)
    )
    expected_dropbacks = (
        float(team_week_dropbacks.mean()) if not team_week_dropbacks.empty else 35.0
    )
    return starter, backup, starter_value, backup_value, expected_dropbacks


def build_qb_replacement_table(
    injuries: pd.DataFrame | None,
    pbp: pd.DataFrame | None,
    rosters: pd.DataFrame | None = None,
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

    qb_groups = _qb_groups(dropbacks)
    roster_groups = _roster_qb_groups(rosters)
    coverage_keys = [
        (int(season), int(week))
        for season, week in covered[["season", "week"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ]
    league_priors = _league_epa_priors(dropbacks, coverage_keys)
    report_groups = {
        (int(season), int(week), str(team)): group
        for (season, week, team), group in reports.groupby(["season", "week", "team"], sort=False)
    }

    rows: list[dict[str, Any]] = []
    for season, week, team in covered.itertuples(index=False, name=None):
        history = _qb_history(
            qb_groups,
            season=int(season),
            week=int(week),
            team=str(team),
            lookback_weeks=lookback_weeks,
        )
        league_prior = league_priors.get((int(season), int(week)), 0.0)
        roster_qbs = _current_roster_qbs(
            roster_groups,
            season=int(season),
            week=int(week),
            team=str(team),
        )
        starter, backup, starter_value, backup_value, expected_dropbacks = _starter_and_backup(
            history,
            dropbacks=dropbacks,
            roster_qbs=roster_qbs,
            season=int(season),
            week=int(week),
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
                severity = max(injury_unavailability_weight(row) for _, row in matching.iterrows())

        value_gap = float(starter_value - backup_value)
        positive_value_gap = max(value_gap, 0.0)
        expected_points_lost = float(severity * positive_value_gap * expected_dropbacks)
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


def attach_qb_shadow_forecasts(
    predictions: list[dict[str, Any]],
    qb_table: pd.DataFrame,
    *,
    injury_snapshot_at: str | None,
    injury_available_week: int | None,
    injury_stale_for_prediction_week: bool,
    shadow_lambda: float = QB_SHADOW_LAMBDA,
) -> list[dict[str, Any]]:
    """Freeze a research-only prospective QB adjustment beside each forecast.

    The shadow forecast never replaces or mutates the published prediction. It
    is emitted only when the injury feed is fresh for the forecast week.
    """
    indexed = None if qb_table.empty else qb_table.set_index(["season", "week", "team"])

    output: list[dict[str, Any]] = []
    for prediction in predictions:
        frozen = dict(prediction)
        season = int(prediction["season"])
        week = int(prediction["week"])
        home_team = str(prediction["home_team"])
        away_team = str(prediction["away_team"])
        football = prediction.get("football_only") or {}
        base_margin = float(football.get("home_margin", prediction["predicted_home_margin"]))

        def team_record(
            team: str,
            *,
            season_key: int = season,
            week_key: int = week,
        ) -> dict[str, Any] | None:
            if indexed is None:
                return None
            key = (season_key, week_key, team)
            if key not in indexed.index:
                return None
            row = indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return {
                "starter_id": row.get("qb_starter_id"),
                "backup_id": row.get("qb_backup_id"),
                "starter_reported": bool(row.get("qb_starter_reported", 0.0)),
                "unavailability_weight": float(row.get("qb_unavailability_weight", 0.0)),
                "starter_epa_per_dropback": float(row.get("qb_starter_epa_per_dropback", 0.0)),
                "backup_epa_per_dropback": float(row.get("qb_backup_epa_per_dropback", 0.0)),
                "value_gap_epa_per_dropback": float(row.get("qb_value_gap_epa_per_dropback", 0.0)),
                "expected_dropbacks": float(row.get("qb_expected_dropbacks", 0.0)),
                "expected_points_lost": float(row.get("qb_expected_points_lost", 0.0)),
            }

        home = team_record(home_team)
        away = team_record(away_team)
        eligible = bool(
            not injury_stale_for_prediction_week and home is not None and away is not None
        )
        raw_adjustment = None
        shadow_margin = None
        if eligible:
            raw_adjustment = float(away["expected_points_lost"] - home["expected_points_lost"])
            shadow_margin = float(base_margin + shadow_lambda * raw_adjustment)

        frozen["qb_shadow"] = {
            "research_only": True,
            "applied_to_published_forecast": False,
            "eligible": eligible,
            "ineligible_reason": (
                None
                if eligible
                else (
                    "stale_injury_feed"
                    if injury_stale_for_prediction_week
                    else "missing_qb_context"
                )
            ),
            "injury_snapshot_at": injury_snapshot_at,
            "injury_available_week": injury_available_week,
            "lambda": float(shadow_lambda),
            "base_independent_margin": base_margin,
            "raw_margin_adjustment": raw_adjustment,
            "shadow_independent_margin": shadow_margin,
            "home": home,
            "away": away,
        }
        output.append(frozen)
    return output
