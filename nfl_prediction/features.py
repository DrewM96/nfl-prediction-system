from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import is_division_game

GAME_FEATURES = [
    "home_points_for_l4",
    "home_points_against_l4",
    "home_yards_l4",
    "home_off_epa_l4",
    "home_def_epa_l4",
    "home_turnovers_l4",
    "home_win_pct_l8",
    "home_pressure_allowed_l4",
    "home_pressure_generated_l4",
    "away_points_for_l4",
    "away_points_against_l4",
    "away_yards_l4",
    "away_off_epa_l4",
    "away_def_epa_l4",
    "away_turnovers_l4",
    "away_win_pct_l8",
    "away_pressure_allowed_l4",
    "away_pressure_generated_l4",
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "division_game",
    "week",
    "home_field",
]


PLAYER_FEATURES = {
    "passing": [
        "passing_yards_l4",
        "passing_yards_l8",
        "attempts_l4",
        "attempts_l8",
        "completion_pct_l4",
        "passing_tds_l4",
        "interceptions_l4",
        "snap_share_l4",
        "opponent_def_epa",
    ],
    "receiving": [
        "receiving_yards_l4",
        "receiving_yards_l8",
        "targets_l4",
        "targets_l8",
        "receptions_l4",
        "catch_rate_l4",
        "yards_per_reception_l4",
        "target_share_l4",
        "snap_share_l4",
        "opponent_def_epa",
    ],
    "receptions": [
        "receptions_l4",
        "receptions_l8",
        "targets_l4",
        "targets_l8",
        "catch_rate_l4",
        "target_share_l4",
        "snap_share_l4",
        "opponent_def_epa",
    ],
    "rushing": [
        "rushing_yards_l4",
        "rushing_yards_l8",
        "carries_l4",
        "carries_l8",
        "yards_per_carry_l4",
        "rushing_tds_l4",
        "snap_share_l4",
        "opponent_def_epa",
    ],
}


DEFAULT_PRIORS = {
    "points_for": 22.5,
    "points_against": 22.5,
    "yards": 350.0,
    "off_epa": 0.0,
    "def_epa": 0.0,
    "turnovers": 1.2,
    "win": 0.5,
    "pressure_allowed": 0.20,
    "pressure_generated": 0.20,
}


@dataclass
class FeatureBuildResult:
    games: pd.DataFrame
    team_snapshot: dict[str, dict[str, float]]
    histories: dict[str, list[dict[str, Any]]]


def _series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in frame:
        return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _rolling_shrunk(
    history: list[dict[str, Any]], key: str, window: int, prior: float, prior_weight: float = 2.0
) -> float:
    observed = [float(row[key]) for row in history[-window:] if pd.notna(row.get(key))]
    if not observed:
        return prior
    return (sum(observed) + prior * prior_weight) / (len(observed) + prior_weight)


def _team_state(history: list[dict[str, Any]], priors: dict[str, float]) -> dict[str, float]:
    return {
        "points_for_l4": _rolling_shrunk(history, "points_for", 4, priors["points_for"]),
        "points_against_l4": _rolling_shrunk(
            history, "points_against", 4, priors["points_against"]
        ),
        "yards_l4": _rolling_shrunk(history, "yards", 4, priors["yards"]),
        "off_epa_l4": _rolling_shrunk(history, "off_epa", 4, priors["off_epa"]),
        "def_epa_l4": _rolling_shrunk(history, "def_epa", 4, priors["def_epa"]),
        "turnovers_l4": _rolling_shrunk(history, "turnovers", 4, priors["turnovers"]),
        "win_pct_l8": _rolling_shrunk(history, "win", 8, priors["win"]),
        "pressure_allowed_l4": _rolling_shrunk(
            history, "pressure_allowed", 4, priors["pressure_allowed"]
        ),
        "pressure_generated_l4": _rolling_shrunk(
            history, "pressure_generated", 4, priors["pressure_generated"]
        ),
    }


def _regular_season(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "game_type" in result:
        result = result[result["game_type"].eq("REG")]
    elif "season_type" in result:
        result = result[result["season_type"].eq("REG")]
    return result


def _game_team_summaries(pbp: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    plays = _regular_season(pbp)
    if plays.empty:
        return {}

    plays = plays.copy()
    play_type = plays.get("play_type", pd.Series("", index=plays.index)).fillna("")
    scrimmage = play_type.isin(["pass", "run"])
    dropback = _series(plays, "qb_dropback").eq(1)
    pressure = _series(plays, "qb_hit").eq(1) | _series(plays, "sack").eq(1)
    plays["_scrimmage_epa"] = _series(plays, "epa").where(scrimmage)
    plays["_turnover"] = _series(plays, "interception") + _series(plays, "fumble_lost")
    plays["_dropback"] = dropback.astype(float)
    plays["_pressure"] = (dropback & pressure).astype(float)

    summaries: dict[tuple[str, str], dict[str, float]] = {}
    for (game_id, team), offense in plays.dropna(subset=["game_id", "posteam"]).groupby(
        ["game_id", "posteam"], sort=False
    ):
        defense = plays[(plays["game_id"] == game_id) & (plays.get("defteam") == team)]
        offense_dropbacks = float(offense["_dropback"].sum())
        defense_dropbacks = float(defense["_dropback"].sum())
        summaries[(str(game_id), str(team))] = {
            "yards": float(_series(offense, "yards_gained").sum()),
            "off_epa": float(offense["_scrimmage_epa"].mean()),
            "def_epa": float(defense["_scrimmage_epa"].mean()),
            "turnovers": float(offense["_turnover"].sum()),
            "pressure_allowed": (
                float(offense["_pressure"].sum()) / offense_dropbacks
                if offense_dropbacks
                else np.nan
            ),
            "pressure_generated": (
                float(defense["_pressure"].sum()) / defense_dropbacks
                if defense_dropbacks
                else np.nan
            ),
        }
    return summaries


def build_point_in_time_game_features(
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    include_unplayed: bool = False,
) -> FeatureBuildResult:
    """Build leak-free game rows using only prior calendar dates.

    All games on the same date are featurized before any result from that date
    is added to history. This conservatively prevents Sunday early-window
    outcomes from influencing Sunday late-window predictions.
    """
    schedule = _regular_season(schedules)
    schedule = schedule.copy()
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    schedule = schedule.dropna(subset=["gameday", "home_team", "away_team"])
    schedule = schedule.sort_values(["gameday", "gametime" if "gametime" in schedule else "week"])
    summaries = _game_team_summaries(pbp)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for game_day, day_games in schedule.groupby(schedule["gameday"].dt.date, sort=True):
        pending_updates: list[tuple[str, dict[str, Any]]] = []
        completed_before_day = [item for history in histories.values() for item in history]
        priors = DEFAULT_PRIORS.copy()
        for key in priors:
            values = [float(item[key]) for item in completed_before_day if pd.notna(item.get(key))]
            if values:
                priors[key] = float(np.mean(values))

        for _, game in day_games.iterrows():
            completed = pd.notna(game.get("home_score")) and pd.notna(game.get("away_score"))
            if not completed and not include_unplayed:
                continue
            home = str(game["home_team"])
            away = str(game["away_team"])
            home_state = _team_state(histories[home], priors)
            away_state = _team_state(histories[away], priors)
            home_last = histories[home][-1]["gameday"] if histories[home] else None
            away_last = histories[away][-1]["gameday"] if histories[away] else None
            home_rest = min((game_day - home_last).days, 21) if home_last else 7
            away_rest = min((game_day - away_last).days, 21) if away_last else 7
            neutral = str(game.get("location", "Home")).lower() == "neutral"

            row: dict[str, Any] = {
                "game_id": str(game.get("game_id", "")),
                "season": int(game["season"]),
                "week": int(game["week"]),
                "gameday": pd.Timestamp(game["gameday"]),
                "gametime": str(game.get("gametime", "")),
                "home_team": home,
                "away_team": away,
                "home_rest_days": float(home_rest),
                "away_rest_days": float(away_rest),
                "rest_advantage": float(home_rest - away_rest),
                "division_game": float(is_division_game(home, away)),
                "home_field": 0.0 if neutral else 1.0,
            }
            row.update({f"home_{key}": value for key, value in home_state.items()})
            row.update({f"away_{key}": value for key, value in away_state.items()})
            if completed:
                home_score = float(game["home_score"])
                away_score = float(game["away_score"])
                row.update(
                    {
                        "home_score": home_score,
                        "away_score": away_score,
                        "total_points": home_score + away_score,
                        "home_margin": home_score - away_score,
                    }
                )
            rows.append(row)

            if completed:
                game_id = str(game.get("game_id", ""))
                home_summary = summaries.get((game_id, home), {})
                away_summary = summaries.get((game_id, away), {})
                home_score = float(game["home_score"])
                away_score = float(game["away_score"])
                pending_updates.extend(
                    [
                        (
                            home,
                            {
                                "gameday": game_day,
                                "points_for": home_score,
                                "points_against": away_score,
                                "win": 1.0
                                if home_score > away_score
                                else (0.5 if home_score == away_score else 0.0),
                                **{
                                    key: home_summary.get(key, priors[key])
                                    for key in priors
                                    if key not in {"points_for", "points_against", "win"}
                                },
                            },
                        ),
                        (
                            away,
                            {
                                "gameday": game_day,
                                "points_for": away_score,
                                "points_against": home_score,
                                "win": 1.0
                                if away_score > home_score
                                else (0.5 if away_score == home_score else 0.0),
                                **{
                                    key: away_summary.get(key, priors[key])
                                    for key in priors
                                    if key not in {"points_for", "points_against", "win"}
                                },
                            },
                        ),
                    ]
                )
        for team, update in pending_updates:
            histories[team].append(update)

    team_snapshot = {
        team: _team_state(history, DEFAULT_PRIORS) for team, history in histories.items()
    }
    return FeatureBuildResult(pd.DataFrame(rows), team_snapshot, dict(histories))


def _player_identity(frame: pd.DataFrame, prefix: str) -> tuple[pd.Series, pd.Series]:
    id_column = f"{prefix}_player_id"
    name_column = f"{prefix}_player_name"
    identifiers = frame.get(
        id_column, frame.get(name_column, pd.Series(index=frame.index, dtype=object))
    )
    names = frame.get(name_column, identifiers)
    return identifiers, names


def build_player_game_logs(pbp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    plays = _regular_season(pbp).copy()
    outputs: dict[str, pd.DataFrame] = {}

    passer_id, passer_name = _player_identity(plays, "passer")
    pass_mask = _series(plays, "pass_attempt").eq(1) & passer_id.notna()
    passing = plays.loc[pass_mask].copy()
    passing["player_id"] = passer_id.loc[pass_mask].astype(str)
    passing["player_name"] = passer_name.loc[pass_mask].astype(str)
    passing["passing_yards_value"] = _series(passing, "passing_yards")
    if "passing_yards" not in passing:
        passing["passing_yards_value"] = _series(passing, "yards_gained").where(
            _series(passing, "complete_pass").eq(1), 0.0
        )
    passing["attempt_value"] = _series(passing, "pass_attempt")
    passing["completion_value"] = _series(passing, "complete_pass")
    passing["passing_td_value"] = _series(passing, "pass_touchdown")
    passing["interception_value"] = _series(passing, "interception")
    outputs["passing"] = passing.groupby(
        [
            "season",
            "week",
            "game_id",
            "game_date",
            "posteam",
            "defteam",
            "player_id",
            "player_name",
        ],
        as_index=False,
    ).agg(
        passing_yards=("passing_yards_value", "sum"),
        attempts=("attempt_value", "sum"),
        completions=("completion_value", "sum"),
        passing_tds=("passing_td_value", "sum"),
        interceptions=("interception_value", "sum"),
    )

    receiver_id, receiver_name = _player_identity(plays, "receiver")
    target_mask = _series(plays, "pass_attempt").eq(1) & receiver_id.notna()
    receiving = plays.loc[target_mask].copy()
    receiving["player_id"] = receiver_id.loc[target_mask].astype(str)
    receiving["player_name"] = receiver_name.loc[target_mask].astype(str)
    receiving["target_value"] = 1.0
    receiving["reception_value"] = _series(receiving, "complete_pass")
    receiving["receiving_yards_value"] = _series(receiving, "receiving_yards")
    if "receiving_yards" not in receiving:
        receiving["receiving_yards_value"] = _series(receiving, "yards_gained").where(
            receiving["reception_value"].eq(1), 0.0
        )
    receiving["receiving_td_value"] = _series(receiving, "pass_touchdown")
    team_targets = receiving.groupby(["season", "game_id", "posteam"])["target_value"].transform(
        "sum"
    )
    receiving["team_targets"] = team_targets
    outputs["receiving"] = receiving.groupby(
        [
            "season",
            "week",
            "game_id",
            "game_date",
            "posteam",
            "defteam",
            "player_id",
            "player_name",
        ],
        as_index=False,
    ).agg(
        receiving_yards=("receiving_yards_value", "sum"),
        targets=("target_value", "sum"),
        receptions=("reception_value", "sum"),
        receiving_tds=("receiving_td_value", "sum"),
        team_targets=("team_targets", "first"),
    )

    rusher_id, rusher_name = _player_identity(plays, "rusher")
    rush_mask = _series(plays, "rush_attempt").eq(1) & rusher_id.notna()
    rush_mask &= ~_series(plays, "qb_kneel").eq(1) & ~_series(plays, "qb_spike").eq(1)
    rushing = plays.loc[rush_mask].copy()
    rushing["player_id"] = rusher_id.loc[rush_mask].astype(str)
    rushing["player_name"] = rusher_name.loc[rush_mask].astype(str)
    rushing["rushing_yards_value"] = _series(rushing, "rushing_yards")
    if "rushing_yards" not in rushing:
        rushing["rushing_yards_value"] = _series(rushing, "yards_gained")
    rushing["carry_value"] = 1.0
    rushing["rushing_td_value"] = _series(rushing, "rush_touchdown")
    outputs["rushing"] = rushing.groupby(
        [
            "season",
            "week",
            "game_id",
            "game_date",
            "posteam",
            "defteam",
            "player_id",
            "player_name",
        ],
        as_index=False,
    ).agg(
        rushing_yards=("rushing_yards_value", "sum"),
        carries=("carry_value", "sum"),
        rushing_tds=("rushing_td_value", "sum"),
    )
    return outputs


def add_shifted_rolling_features(
    frame: pd.DataFrame,
    stat_columns: Iterable[str],
    *,
    windows: tuple[int, ...] = (4, 8),
) -> pd.DataFrame:
    result = frame.copy()
    result["game_date"] = pd.to_datetime(result["game_date"], errors="coerce")
    result = result.sort_values(["player_id", "game_date", "game_id"])

    def weighted(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        if not len(clean):
            return np.nan
        weights = np.exp(np.linspace(-1.0, 0.0, len(clean)))
        return float(np.average(clean, weights=weights))

    for stat in stat_columns:
        numeric = pd.to_numeric(result[stat], errors="coerce")
        for window in windows:
            result[f"{stat}_l{window}"] = numeric.groupby(result["player_id"]).transform(
                lambda series, rolling_window=window: series.shift(1)
                .rolling(rolling_window, min_periods=1)
                .apply(weighted, raw=True)
            )
    return result
