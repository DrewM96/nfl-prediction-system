from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

INJURY_STATUS_WEIGHTS = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.35,
}
PRACTICE_STATUS_WEIGHTS = {
    "did not participate in practice": 0.25,
    "limited participation in practice": 0.10,
    "full participation in practice": 0.0,
}

_OFFENSIVE_LINE = {"OL", "T", "OT", "G", "OG", "C"}
_SKILL = {"WR", "TE", "RB", "FB"}
_FRONT = {"DL", "DE", "DT", "NT", "LB", "ILB", "OLB", "EDGE"}
_SECONDARY = {"CB", "S", "DB", "FS", "SS"}

INJURY_TEAM_FEATURES = [
    "injury_reported_count",
    "injury_out_count",
    "injury_doubtful_count",
    "injury_questionable_count",
    "injury_offense_snap_loss",
    "injury_defense_snap_loss",
    "injury_qb_snap_loss",
    "injury_ol_snap_loss",
    "injury_skill_snap_loss",
    "injury_front_snap_loss",
    "injury_secondary_snap_loss",
]


def _game_features(*team_features: str) -> list[str]:
    return [f"{side}_{feature}" for side in ("home", "away") for feature in team_features]


INJURY_CANDIDATE_GAME_FEATURE_GROUPS = {
    "injury_status": _game_features(
        "injury_out_count",
        "injury_doubtful_count",
        "injury_questionable_count",
    ),
    "injury_snap_loss": _game_features(
        "injury_offense_snap_loss",
        "injury_defense_snap_loss",
    ),
    "injury_positions": _game_features(
        "injury_qb_snap_loss",
        "injury_ol_snap_loss",
        "injury_skill_snap_loss",
        "injury_front_snap_loss",
        "injury_secondary_snap_loss",
    ),
}


def _normalized(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def injury_unavailability_weight(row: pd.Series | dict[str, Any]) -> float:
    """Map an official designation to a conservative research-only severity weight.

    Final report status is preferred. Practice participation is used only when a
    game-status designation is unavailable. These are transparent proxy weights,
    not production point adjustments.
    """
    report_status = _normalized(row.get("report_status"))
    if report_status in INJURY_STATUS_WEIGHTS:
        return INJURY_STATUS_WEIGHTS[report_status]
    practice_status = _normalized(row.get("practice_status"))
    return PRACTICE_STATUS_WEIGHTS.get(practice_status, 0.0)


def _player_crosswalk(rosters: pd.DataFrame) -> dict[tuple[int, str], str]:
    required = {"season", "gsis_id", "pfr_id"}
    if rosters.empty or not required.issubset(rosters):
        return {}
    frame = rosters[list(required)].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["gsis_id"] = frame["gsis_id"].fillna("").astype(str).str.strip()
    frame["pfr_id"] = frame["pfr_id"].fillna("").astype(str).str.strip()
    frame = frame[
        frame["season"].notna() & frame["gsis_id"].ne("") & frame["pfr_id"].ne("")
    ].drop_duplicates(["season", "gsis_id"], keep="last")
    return {
        (int(row["season"]), str(row["gsis_id"])): str(row["pfr_id"])
        for _, row in frame.iterrows()
    }


def _prepare_snap_counts(snap_counts: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "week",
        "team",
        "pfr_player_id",
        "position",
        "offense_snaps",
        "defense_snaps",
    }
    if snap_counts.empty or not required.issubset(snap_counts):
        return pd.DataFrame(columns=sorted(required))
    frame = snap_counts.copy()
    if "game_type" in frame:
        frame = frame[frame["game_type"].eq("REG")]
    for column in ("season", "week", "offense_snaps", "defense_snaps"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["season"].notna() & frame["week"].notna()].copy()
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["team"] = frame["team"].fillna("").astype(str)
    frame["pfr_player_id"] = frame["pfr_player_id"].fillna("").astype(str)
    frame["position"] = frame["position"].fillna("").astype(str).str.upper()
    frame["offense_snaps"] = frame["offense_snaps"].fillna(0.0)
    frame["defense_snaps"] = frame["defense_snaps"].fillna(0.0)
    return frame


def _prior_snap_share(
    snaps: pd.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    pfr_player_id: str,
    side: str,
    lookback_weeks: int,
) -> float:
    snap_column = f"{side}_snaps"
    current = snaps[
        snaps["season"].eq(season)
        & snaps["team"].eq(team)
        & snaps["week"].lt(week)
    ]
    if not current.empty:
        eligible_weeks = sorted(current["week"].unique())[-lookback_weeks:]
        current = current[current["week"].isin(eligible_weeks)]
    else:
        current = snaps[
            snaps["season"].eq(season - 1)
            & snaps["team"].eq(team)
        ]
        if not current.empty:
            eligible_weeks = sorted(current["week"].unique())[-lookback_weeks:]
            current = current[current["week"].isin(eligible_weeks)]
    if current.empty:
        return 0.0

    # Team plays are represented by the largest individual snap count in each
    # game/week. Summing all player snaps would multiply the denominator by 11.
    team_week_snaps = current.groupby("week")[snap_column].max().sum()
    if team_week_snaps <= 0:
        return 0.0
    player_snaps = current.loc[
        current["pfr_player_id"].eq(pfr_player_id), snap_column
    ].sum()
    return float(np.clip(player_snaps / team_week_snaps, 0.0, 1.0))


def _position_bucket(position: str) -> str | None:
    position = position.upper()
    if position == "QB":
        return "qb"
    if position in _OFFENSIVE_LINE:
        return "ol"
    if position in _SKILL:
        return "skill"
    if position in _FRONT:
        return "front"
    if position in _SECONDARY:
        return "secondary"
    return None


def build_injury_availability_table(
    injuries: pd.DataFrame | None,
    snap_counts: pd.DataFrame | None,
    rosters: pd.DataFrame | None,
    *,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """Build team-week injury proxies using only snaps from prior games.

    The nflverse historical injury table contains weekly report/practice status,
    not a timestamped sequence of every intrawEEK publication. Therefore this is
    a late-week/final-report research proxy and must not be interpreted as what a
    Tuesday forecast knew.
    """
    injuries = injuries if injuries is not None else pd.DataFrame()
    snaps = _prepare_snap_counts(snap_counts if snap_counts is not None else pd.DataFrame())
    rosters = rosters if rosters is not None else pd.DataFrame()
    required = {"season", "week", "team"}
    if injuries.empty or not required.issubset(injuries):
        return pd.DataFrame(columns=["season", "week", "team", *INJURY_TEAM_FEATURES])

    frame = injuries.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame[frame["season"].notna() & frame["week"].notna()].copy()
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["team"] = frame["team"].fillna("").astype(str)
    frame["gsis_id"] = frame.get("gsis_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["position"] = (
        frame.get("position", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    )
    frame = frame[
        frame.get("report_status", pd.Series(index=frame.index, dtype=object)).notna()
        | frame.get("practice_status", pd.Series(index=frame.index, dtype=object)).notna()
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=["season", "week", "team", *INJURY_TEAM_FEATURES])

    crosswalk = _player_crosswalk(rosters)
    rows: list[dict[str, Any]] = []
    for _, injury in frame.iterrows():
        season = int(injury["season"])
        week = int(injury["week"])
        team = str(injury["team"])
        gsis_id = str(injury.get("gsis_id") or "")
        pfr_id = crosswalk.get((season, gsis_id), "")
        severity = injury_unavailability_weight(injury)
        offense_share = (
            _prior_snap_share(
                snaps,
                season=season,
                week=week,
                team=team,
                pfr_player_id=pfr_id,
                side="offense",
                lookback_weeks=lookback_weeks,
            )
            if pfr_id
            else 0.0
        )
        defense_share = (
            _prior_snap_share(
                snaps,
                season=season,
                week=week,
                team=team,
                pfr_player_id=pfr_id,
                side="defense",
                lookback_weeks=lookback_weeks,
            )
            if pfr_id
            else 0.0
        )
        status = _normalized(injury.get("report_status"))
        bucket = _position_bucket(str(injury.get("position") or ""))
        rows.append(
            {
                "season": season,
                "week": week,
                "team": team,
                "injury_reported_count": 1.0,
                "injury_out_count": float(status == "out"),
                "injury_doubtful_count": float(status == "doubtful"),
                "injury_questionable_count": float(status == "questionable"),
                "injury_offense_snap_loss": severity * offense_share,
                "injury_defense_snap_loss": severity * defense_share,
                "injury_qb_snap_loss": severity * offense_share if bucket == "qb" else 0.0,
                "injury_ol_snap_loss": severity * offense_share if bucket == "ol" else 0.0,
                "injury_skill_snap_loss": severity * offense_share if bucket == "skill" else 0.0,
                "injury_front_snap_loss": severity * defense_share if bucket == "front" else 0.0,
                "injury_secondary_snap_loss": (
                    severity * defense_share if bucket == "secondary" else 0.0
                ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["season", "week", "team", *INJURY_TEAM_FEATURES])
    result = pd.DataFrame(rows)
    return (
        result.groupby(["season", "week", "team"], as_index=False)[INJURY_TEAM_FEATURES]
        .sum()
        .sort_values(["season", "week", "team"])
        .reset_index(drop=True)
    )


def attach_injury_availability_features(
    games: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    result = games.copy()
    for side in ("home", "away"):
        for feature in INJURY_TEAM_FEATURES:
            result[f"{side}_{feature}"] = 0.0
    if result.empty or availability.empty:
        return result

    indexed = availability.set_index(["season", "week", "team"])
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
            for feature in INJURY_TEAM_FEATURES:
                result.at[index, f"{side}_{feature}"] = float(row[feature])
    return result


def injury_coverage_weeks(injuries: pd.DataFrame | None) -> set[tuple[int, int]]:
    if injuries is None or injuries.empty or not {"season", "week"}.issubset(injuries):
        return set()
    frame = injuries[["season", "week"]].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna().astype(int)
    return set(map(tuple, frame[["season", "week"]].drop_duplicates().to_numpy()))


def filter_to_injury_covered_games(
    games: pd.DataFrame,
    injuries: pd.DataFrame | None,
) -> pd.DataFrame:
    covered = injury_coverage_weeks(injuries)
    if not covered:
        return games.iloc[0:0].copy()
    mask = [
        (int(season), int(week)) in covered
        for season, week in zip(games["season"], games["week"], strict=False)
    ]
    return games.loc[mask].copy()
