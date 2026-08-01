from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

ROSTER_RAW_FEATURES = [
    "roster_offense_continuity",
    "roster_defense_continuity",
    "roster_qb_returning",
    "roster_ol_continuity",
    "roster_skill_continuity",
    "roster_front_continuity",
    "roster_secondary_continuity",
    "roster_incoming_offense_share",
    "roster_incoming_defense_share",
]
ROSTER_TEAM_FEATURES = [f"{feature}_delta" for feature in ROSTER_RAW_FEATURES]


def _game_features(*team_features: str) -> list[str]:
    return [f"{side}_{feature}" for side in ("home", "away") for feature in team_features]


ROSTER_CANDIDATE_GAME_FEATURE_GROUPS = {
    "roster_overall": _game_features(
        "roster_offense_continuity_delta",
        "roster_defense_continuity_delta",
    ),
    "roster_offense_positions": _game_features(
        "roster_qb_returning_delta",
        "roster_ol_continuity_delta",
        "roster_skill_continuity_delta",
    ),
    "roster_defense_positions": _game_features(
        "roster_front_continuity_delta",
        "roster_secondary_continuity_delta",
    ),
    "roster_veteran_additions": _game_features(
        "roster_incoming_offense_share_delta",
        "roster_incoming_defense_share_delta",
    ),
}

_DEPARTED_STATUSES = {"CUT", "RET", "TRD", "TRC"}
_OFFENSIVE_LINE = {"OL", "T", "OT", "G", "C"}
_SKILL = {"WR", "TE", "RB", "FB"}
_FRONT = {"DL", "DE", "DT", "NT", "LB", "ILB", "OLB", "EDGE"}
_SECONDARY = {"CB", "S", "DB", "FS", "SS"}


def _name_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    normalized = re.sub(r"[^a-z0-9]", "", ascii_name.lower())
    return f"name:{normalized}" if normalized else ""


def roster_transition_weight(week: int | float) -> float:
    """Return the preseason-prior weight, fully tapered after Week 4."""
    numeric_week = max(float(week), 1.0)
    return float(np.clip(1.25 - 0.25 * numeric_week, 0.0, 1.0))


def decay_roster_feature(feature: str, value: float, week: int | float) -> float:
    if "roster_" not in feature:
        return float(value)
    return float(value) * roster_transition_weight(week)


def _opening_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "team"}
    if (
        rosters.empty
        or not required.issubset(rosters)
        or not ({"pfr_id", "full_name"} & set(rosters))
    ):
        return pd.DataFrame(columns=["season", "team", "player_id"])
    frame = rosters.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame.get("week", 1), errors="coerce").fillna(1)
    if "game_type" in frame:
        regular = frame[frame["game_type"].eq("REG")]
        if not regular.empty:
            frame = regular
    opening_weeks = frame.groupby("season")["week"].transform("min")
    frame = frame[frame["week"].eq(opening_weeks)]
    if "status" in frame:
        frame = frame[~frame["status"].fillna("").isin(_DEPARTED_STATUSES)]
    frame_pfr = frame.get("pfr_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame_names = frame.get("full_name", pd.Series("", index=frame.index)).map(_name_key)
    frame["player_id"] = frame_names.where(frame_names.ne(""), "pfr:" + frame_pfr.str.strip())
    frame.loc[frame_pfr.str.strip().eq("") & frame_names.eq(""), "player_id"] = ""
    frame = frame[frame["player_id"].ne("")]
    return frame[["season", "team", "player_id"]].drop_duplicates()


def _player_snaps(snap_counts: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "team",
        "pfr_player_id",
        "position",
        "offense_snaps",
        "defense_snaps",
    }
    if snap_counts.empty or not required.issubset(snap_counts):
        return pd.DataFrame(
            columns=[
                "season",
                "team",
                "player_id",
                "position",
                "offense_snaps",
                "defense_snaps",
            ]
        )
    frame = snap_counts.copy()
    if "game_type" in frame:
        frame = frame[frame["game_type"].eq("REG")]
    pfr_ids = frame["pfr_player_id"].fillna("").astype(str).str.strip()
    player_names = frame.get("player", pd.Series("", index=frame.index)).map(_name_key)
    frame["player_id"] = player_names.where(player_names.ne(""), "pfr:" + pfr_ids)
    frame.loc[pfr_ids.eq("") & player_names.eq(""), "player_id"] = ""
    frame = frame[frame["player_id"].ne("")]
    frame["position"] = frame["position"].fillna("").astype(str).str.upper()
    for column in ("offense_snaps", "defense_snaps"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return (
        frame.groupby(["season", "team", "player_id", "position"], as_index=False)[
            ["offense_snaps", "defense_snaps"]
        ]
        .sum()
        .reset_index(drop=True)
    )


def _share(
    prior_team: pd.DataFrame,
    current_ids: set[str],
    *,
    snap_column: str,
    positions: set[str] | None = None,
) -> float:
    eligible = (
        prior_team if positions is None else prior_team[prior_team["position"].isin(positions)]
    )
    denominator = float(eligible[snap_column].sum())
    if denominator <= 0:
        return np.nan
    returning = eligible[eligible["player_id"].isin(current_ids)]
    return float(returning[snap_column].sum()) / denominator


def _incoming_share(
    prior_all: pd.DataFrame,
    prior_team: pd.DataFrame,
    current_ids: set[str],
    team: str,
    *,
    snap_column: str,
) -> float:
    denominator = float(prior_team[snap_column].sum())
    if denominator <= 0:
        return np.nan
    incoming = prior_all[prior_all["player_id"].isin(current_ids) & prior_all["team"].ne(team)]
    return float(np.clip(incoming[snap_column].sum() / denominator, 0.0, 1.0))


def build_roster_transition_table(
    rosters: pd.DataFrame | None,
    snap_counts: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build season-opening roster changes using prior-year regular-season snaps.

    Opening rosters determine team membership only. Game-day active/inactive status
    is intentionally ignored so a historical Week 1 inactive list cannot leak into
    an earlier preseason forecast.
    """
    opening = _opening_rosters(rosters if rosters is not None else pd.DataFrame())
    snaps = _player_snaps(snap_counts if snap_counts is not None else pd.DataFrame())
    if opening.empty or snaps.empty:
        return pd.DataFrame(columns=["season", "team", *ROSTER_RAW_FEATURES, *ROSTER_TEAM_FEATURES])

    rows: list[dict[str, Any]] = []
    for (season, team), current in opening.groupby(["season", "team"], sort=True):
        prior_all = snaps[snaps["season"].eq(int(season) - 1)]
        prior_team = prior_all[prior_all["team"].eq(team)]
        if prior_team.empty:
            continue
        current_ids = set(current["player_id"])
        quarterbacks = prior_team[
            prior_team["position"].eq("QB") & prior_team["offense_snaps"].gt(0)
        ]
        primary_qb = (
            str(quarterbacks.loc[quarterbacks["offense_snaps"].idxmax(), "player_id"])
            if not quarterbacks.empty
            else None
        )
        rows.append(
            {
                "season": int(season),
                "team": str(team),
                "roster_offense_continuity": _share(
                    prior_team, current_ids, snap_column="offense_snaps"
                ),
                "roster_defense_continuity": _share(
                    prior_team, current_ids, snap_column="defense_snaps"
                ),
                "roster_qb_returning": float(primary_qb in current_ids) if primary_qb else np.nan,
                "roster_ol_continuity": _share(
                    prior_team,
                    current_ids,
                    snap_column="offense_snaps",
                    positions=_OFFENSIVE_LINE,
                ),
                "roster_skill_continuity": _share(
                    prior_team,
                    current_ids,
                    snap_column="offense_snaps",
                    positions=_SKILL,
                ),
                "roster_front_continuity": _share(
                    prior_team,
                    current_ids,
                    snap_column="defense_snaps",
                    positions=_FRONT,
                ),
                "roster_secondary_continuity": _share(
                    prior_team,
                    current_ids,
                    snap_column="defense_snaps",
                    positions=_SECONDARY,
                ),
                "roster_incoming_offense_share": _incoming_share(
                    prior_all,
                    prior_team,
                    current_ids,
                    str(team),
                    snap_column="offense_snaps",
                ),
                "roster_incoming_defense_share": _incoming_share(
                    prior_all,
                    prior_team,
                    current_ids,
                    str(team),
                    snap_column="defense_snaps",
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["season", "team", *ROSTER_RAW_FEATURES, *ROSTER_TEAM_FEATURES])
    for feature in ROSTER_RAW_FEATURES:
        season_mean = result.groupby("season")[feature].transform("mean")
        result[feature] = result[feature].fillna(season_mean)
        result[f"{feature}_delta"] = (result[feature] - season_mean).fillna(0.0)
    return result


def attach_roster_transition_features(
    games: pd.DataFrame,
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    result = games.copy()
    for side in ("home", "away"):
        for feature in ROSTER_TEAM_FEATURES:
            result[f"{side}_{feature}"] = 0.0
    if result.empty or transitions.empty:
        return result

    indexed = transitions.set_index(["season", "team"])
    for index, game in result.iterrows():
        decay = roster_transition_weight(game["week"])
        for side in ("home", "away"):
            key = (int(game["season"]), str(game[f"{side}_team"]))
            if key not in indexed.index:
                continue
            transition = indexed.loc[key]
            for feature in ROSTER_TEAM_FEATURES:
                result.at[index, f"{side}_{feature}"] = float(transition[feature]) * decay
    return result


def roster_snapshot_for_season(
    transitions: pd.DataFrame,
    season: int,
) -> dict[str, dict[str, float]]:
    if transitions.empty:
        return {}
    current = transitions[transitions["season"].eq(season)]
    return {
        str(row["team"]): {
            feature: float(row[feature])
            for feature in [*ROSTER_RAW_FEATURES, *ROSTER_TEAM_FEATURES]
        }
        for _, row in current.iterrows()
    }
