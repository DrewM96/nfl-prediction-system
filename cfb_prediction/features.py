from __future__ import annotations

from collections import defaultdict
from math import log
from typing import Any

import numpy as np
import pandas as pd

from .data import CFBHistoricalData

PRESEASON_TEAM_FEATURES = [
    "returning_ppa_z",
    "returning_passing_ppa_z",
    "returning_rushing_ppa_z",
    "returning_receiving_ppa_z",
    "talent_z",
    "recruiting_points_z",
    "portal_net_rating_z",
]

DYNAMIC_KEYS = [
    "points_for",
    "points_against",
    "margin_vs_elo",
    "opponent_elo",
    "off_ppa",
    "def_ppa",
    "off_success_rate",
    "def_success_rate",
    "off_explosiveness",
    "def_explosiveness",
    "plays",
]

PRIORS = {
    "points_for": 27.0,
    "points_against": 27.0,
    "margin_vs_elo": 0.0,
    "opponent_elo": 1500.0,
    "off_ppa": 0.0,
    "def_ppa": 0.0,
    "off_success_rate": 0.42,
    "def_success_rate": 0.42,
    "off_explosiveness": 1.2,
    "def_explosiveness": 1.2,
    "plays": 70.0,
}

CFB_ELO_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_expected_margin",
    "week",
    "home_field",
    "conference_game",
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
]
CFB_FORM_FEATURES = CFB_ELO_FEATURES + [
    f"{side}_{key}_l6"
    for side in ("home", "away")
    for key in (
        "points_for",
        "points_against",
        "margin_vs_elo",
        "opponent_elo",
        "plays",
    )
]
CFB_ADVANCED_FEATURES = CFB_FORM_FEATURES + [
    f"{side}_{key}_l6"
    for side in ("home", "away")
    for key in (
        "off_ppa",
        "def_ppa",
        "off_success_rate",
        "def_success_rate",
        "off_explosiveness",
        "def_explosiveness",
    )
]
CFB_FULL_FEATURES = CFB_ADVANCED_FEATURES + [
    f"{side}_{feature}" for side in ("home", "away") for feature in PRESEASON_TEAM_FEATURES
]
CFB_FEATURE_CONFIGURATIONS = {
    "elo_context": CFB_ELO_FEATURES,
    "elo_form": CFB_FORM_FEATURES,
    "elo_form_advanced": CFB_ADVANCED_FEATURES,
    "elo_form_advanced_preseason": CFB_FULL_FEATURES,
}


def _zscore_by_season(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    medians = values.groupby(frame["season"]).transform("median")
    values = values.fillna(medians)
    means = values.groupby(frame["season"]).transform("mean")
    standard_deviations = values.groupby(frame["season"]).transform("std").replace(0, np.nan)
    return ((values - means) / standard_deviations).fillna(0.0)


def build_preseason_table(data: CFBHistoricalData) -> pd.DataFrame:
    home = data.games[["season", "home_team"]].rename(columns={"home_team": "team"})
    away = data.games[["season", "away_team"]].rename(columns={"away_team": "team"})
    teams = pd.concat([home, away], ignore_index=True).drop_duplicates(["season", "team"])

    returning = (
        data.returning.drop_duplicates(["season", "team"], keep="last")
        if {"season", "team"}.issubset(data.returning)
        else pd.DataFrame(columns=["season", "team"])
    )
    talent = (
        data.talent.groupby(["season", "team"], as_index=False)["talent"].last()
        if {"season", "team", "talent"}.issubset(data.talent)
        else pd.DataFrame(columns=["season", "team", "talent"])
    )
    recruiting = (
        data.recruiting.groupby(["season", "team"], as_index=False)["recruiting_points"].last()
        if {"season", "team", "recruiting_points"}.issubset(data.recruiting)
        else pd.DataFrame(columns=["season", "team", "recruiting_points"])
    )

    portal = data.portal.copy()
    for column, default in (
        ("season", pd.NA),
        ("origin", ""),
        ("destination", ""),
        ("rating", 0.0),
    ):
        if column not in portal:
            portal[column] = default
    portal["rating"] = pd.to_numeric(portal["rating"], errors="coerce").fillna(0.0)
    season_starts = data.games.groupby("season")["start_date"].min()
    portal["season_start"] = portal["season"].map(season_starts)
    portal = portal[
        portal["transfer_date"].notna()
        & portal["season_start"].notna()
        & portal["transfer_date"].lt(portal["season_start"])
        & portal["transfer_date"].ge(portal["season_start"] - pd.Timedelta(days=400))
    ]
    incoming = (
        portal[portal["destination"].fillna("").ne("")]
        .groupby(["season", "destination"], as_index=False)["rating"]
        .sum()
        .rename(columns={"destination": "team", "rating": "portal_in_rating"})
    )
    outgoing = (
        portal[portal["origin"].fillna("").ne("")]
        .groupby(["season", "origin"], as_index=False)["rating"]
        .sum()
        .rename(columns={"origin": "team", "rating": "portal_out_rating"})
    )

    result = teams.merge(returning, on=["season", "team"], how="left")
    result = result.merge(talent, on=["season", "team"], how="left")
    result = result.merge(recruiting, on=["season", "team"], how="left")
    result = result.merge(incoming, on=["season", "team"], how="left")
    result = result.merge(outgoing, on=["season", "team"], how="left")
    result["portal_net_rating"] = result["portal_in_rating"].fillna(0.0) - result[
        "portal_out_rating"
    ].fillna(0.0)

    raw_to_feature = {
        "returning_ppa": "returning_ppa_z",
        "returning_passing_ppa": "returning_passing_ppa_z",
        "returning_rushing_ppa": "returning_rushing_ppa_z",
        "returning_receiving_ppa": "returning_receiving_ppa_z",
        "talent": "talent_z",
        "recruiting_points": "recruiting_points_z",
        "portal_net_rating": "portal_net_rating_z",
    }
    for raw, feature in raw_to_feature.items():
        if raw not in result:
            result[raw] = np.nan
        result[feature] = _zscore_by_season(result, raw)
    return result[["season", "team", *PRESEASON_TEAM_FEATURES]]


def _shrunk_state(history: list[dict[str, float]], season: int) -> dict[str, float]:
    recent = history[-6:]
    result: dict[str, float] = {}
    for key in DYNAMIC_KEYS:
        prior = PRIORS[key]
        numerator = 3.0 * prior
        denominator = 3.0
        for position, record in enumerate(recent, start=1):
            recency = position / max(len(recent), 1)
            offseason = 1.0 if int(record["season"]) == season else 0.35
            weight = recency * offseason
            numerator += weight * float(record.get(key, prior))
            denominator += weight
        result[f"{key}_l6"] = numerator / denominator
    return result


def _advanced_lookup(frame: pd.DataFrame) -> dict[tuple[int, str], dict[str, float]]:
    if frame.empty:
        return {}
    metric_columns = [
        "off_ppa",
        "def_ppa",
        "off_success_rate",
        "def_success_rate",
        "off_explosiveness",
        "def_explosiveness",
        "off_plays",
    ]
    return {
        (int(row["game_id"]), str(row["team"])): {
            column: float(row[column]) if pd.notna(row[column]) else np.nan
            for column in metric_columns
        }
        for _, row in frame.iterrows()
    }


def _safe_metric(record: dict[str, float] | None, key: str, prior_key: str) -> float:
    if not record:
        return PRIORS[prior_key]
    value = record.get(key)
    return PRIORS[prior_key] if value is None or not np.isfinite(value) else float(value)


def _elo_expected(home_elo: float, away_elo: float, home_field: float) -> tuple[float, float]:
    rating_difference = home_elo - away_elo + 55.0 * home_field
    probability = 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))
    expected_margin = rating_difference / 25.0
    return probability, expected_margin


def build_point_in_time_features(
    data: CFBHistoricalData,
    *,
    include_scheduled: bool = False,
) -> pd.DataFrame:
    """Build features using only results available before each game's kickoff.

    Historical benchmarking keeps the default completed-game behavior. Production
    calls may include scheduled games; those rows receive pregame features but
    never update Elo, form, advanced metrics, or last-played state.
    """
    games = data.games.copy()
    eligible = (
        games["fbs_vs_fbs"].fillna(False)
        & games["home_id"].notna()
        & games["away_id"].notna()
        & games["week"].notna()
        & games["start_date"].notna()
    )
    completed = (
        games["completed"].fillna(False)
        & games["home_points"].notna()
        & games["away_points"].notna()
    )
    games = games[eligible & (True if include_scheduled else completed)].copy()
    games = games.sort_values(["season", "week", "start_date", "game_id"], ignore_index=True)
    if games.empty:
        return pd.DataFrame()

    preseason = build_preseason_table(data).set_index(["season", "team"])
    advanced = _advanced_lookup(data.advanced)
    lines = data.lines.set_index("game_id") if not data.lines.empty else pd.DataFrame()
    histories: defaultdict[int, list[dict[str, float]]] = defaultdict(list)
    ratings: defaultdict[int, float] = defaultdict(lambda: 1500.0)
    last_played: dict[int, pd.Timestamp] = {}
    rows: list[dict[str, Any]] = []
    active_season: int | None = None

    for _, game in games.iterrows():
        season = int(game["season"])
        if active_season != season:
            if active_season is not None:
                for team_id in list(ratings):
                    ratings[team_id] = 1500.0 + 0.65 * (ratings[team_id] - 1500.0)
            active_season = season

        home_id = int(game["home_id"])
        away_id = int(game["away_id"])
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        kickoff = pd.Timestamp(game["start_date"])
        is_completed = (
            bool(game["completed"])
            and pd.notna(game["home_points"])
            and pd.notna(game["away_points"])
        )
        home_field = 0.0 if bool(game["neutral_site"]) else 1.0
        home_elo = float(ratings[home_id])
        away_elo = float(ratings[away_id])
        expected_probability, expected_margin = _elo_expected(home_elo, away_elo, home_field)
        home_state = _shrunk_state(histories[home_id], season)
        away_state = _shrunk_state(histories[away_id], season)
        home_rest = (
            min(max((kickoff - last_played[home_id]).days, 4), 28) if home_id in last_played else 7
        )
        away_rest = (
            min(max((kickoff - last_played[away_id]).days, 4), 28) if away_id in last_played else 7
        )

        row: dict[str, Any] = {
            "game_id": int(game["game_id"]),
            "season": season,
            "week": int(game["week"]),
            "start_date": kickoff,
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_expected_margin": expected_margin,
            "home_field": home_field,
            "conference_game": float(bool(game["conference_game"])),
            "home_rest_days": float(home_rest),
            "away_rest_days": float(away_rest),
            "rest_advantage": float(home_rest - away_rest),
            "completed": is_completed,
            "home_margin": (
                float(game["home_points"] - game["away_points"]) if is_completed else np.nan
            ),
            "total_points": (
                float(game["home_points"] + game["away_points"]) if is_completed else np.nan
            ),
            "form_expected_total": 0.5
            * (
                home_state["points_for_l6"]
                + away_state["points_against_l6"]
                + away_state["points_for_l6"]
                + home_state["points_against_l6"]
            ),
        }
        for side, state, team in (
            ("home", home_state, home_team),
            ("away", away_state, away_team),
        ):
            row.update({f"{side}_{key}": value for key, value in state.items()})
            key = (season, team)
            for feature in PRESEASON_TEAM_FEATURES:
                row[f"{side}_{feature}"] = (
                    float(preseason.loc[key, feature]) if key in preseason.index else 0.0
                )
        if not data.lines.empty and int(game["game_id"]) in lines.index:
            market = lines.loc[int(game["game_id"])]
            row["market_home_margin"] = market["market_home_margin"]
            row["market_total"] = market["market_total"]
            row["market_provider_count"] = market["market_provider_count"]
        else:
            row["market_home_margin"] = np.nan
            row["market_total"] = np.nan
            row["market_provider_count"] = 0
        rows.append(row)

        if not is_completed:
            # Scheduled dates are known at forecast time and determine rest for
            # later games; unknown results never update performance state.
            last_played[home_id] = kickoff
            last_played[away_id] = kickoff
            continue

        actual_margin = float(game["home_points"] - game["away_points"])
        home_advanced = advanced.get((int(game["game_id"]), home_team))
        away_advanced = advanced.get((int(game["game_id"]), away_team))
        home_record = {
            "season": float(season),
            "points_for": float(game["home_points"]),
            "points_against": float(game["away_points"]),
            "margin_vs_elo": actual_margin - expected_margin,
            "opponent_elo": away_elo,
            "off_ppa": _safe_metric(home_advanced, "off_ppa", "off_ppa"),
            "def_ppa": _safe_metric(home_advanced, "def_ppa", "def_ppa"),
            "off_success_rate": _safe_metric(home_advanced, "off_success_rate", "off_success_rate"),
            "def_success_rate": _safe_metric(home_advanced, "def_success_rate", "def_success_rate"),
            "off_explosiveness": _safe_metric(
                home_advanced, "off_explosiveness", "off_explosiveness"
            ),
            "def_explosiveness": _safe_metric(
                home_advanced, "def_explosiveness", "def_explosiveness"
            ),
            "plays": _safe_metric(home_advanced, "off_plays", "plays"),
        }
        away_record = {
            "season": float(season),
            "points_for": float(game["away_points"]),
            "points_against": float(game["home_points"]),
            "margin_vs_elo": -actual_margin + expected_margin,
            "opponent_elo": home_elo,
            "off_ppa": _safe_metric(away_advanced, "off_ppa", "off_ppa"),
            "def_ppa": _safe_metric(away_advanced, "def_ppa", "def_ppa"),
            "off_success_rate": _safe_metric(away_advanced, "off_success_rate", "off_success_rate"),
            "def_success_rate": _safe_metric(away_advanced, "def_success_rate", "def_success_rate"),
            "off_explosiveness": _safe_metric(
                away_advanced, "off_explosiveness", "off_explosiveness"
            ),
            "def_explosiveness": _safe_metric(
                away_advanced, "def_explosiveness", "def_explosiveness"
            ),
            "plays": _safe_metric(away_advanced, "off_plays", "plays"),
        }
        histories[home_id].append(home_record)
        histories[away_id].append(away_record)
        last_played[home_id] = kickoff
        last_played[away_id] = kickoff

        result = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)
        margin_multiplier = (
            log(abs(actual_margin) + 1.0) * 2.2 / (abs(home_elo - away_elo) * 0.001 + 2.2)
        )
        adjustment = 20.0 * margin_multiplier * (result - expected_probability)
        ratings[home_id] += adjustment
        ratings[away_id] -= adjustment

    return pd.DataFrame(rows)
