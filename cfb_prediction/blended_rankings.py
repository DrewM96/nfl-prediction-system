from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .data import CFBHistoricalData
from .features import CFB_FULL_FEATURES, build_point_in_time_features


def team_metadata(games: pd.DataFrame, season: int) -> dict[str, dict]:
    metadata = {}
    current = games[games.season.eq(season) & games.fbs_vs_fbs.fillna(False)]
    for _, game in current.iterrows():
        for side in ("home", "away"):
            metadata[str(game[f"{side}_team"])] = {
                "id": int(game[f"{side}_id"]),
                "conference": str(game[f"{side}_conference"]),
            }
    return metadata


def freeze_data(data: CFBHistoricalData, season: int, week: int, cutoff: pd.Timestamp):
    """Remove whole validation-week outcomes and every outcome at/after cutoff."""
    games = data.games[data.games.season.le(season)].copy()
    unknown = games.start_date.ge(cutoff) | (games.season.eq(season) & games.week.ge(week))
    games.loc[unknown, "completed"] = False
    games.loc[unknown, ["home_points", "away_points"]] = np.nan
    known_ids = set(games.loc[games.completed.fillna(False), "game_id"].astype(int))
    advanced = data.advanced[data.advanced.game_id.isin(known_ids)].copy()
    return replace(data, games=games, advanced=advanced)


def frozen_frames(data, season, week, cutoff, feature_names=CFB_FULL_FEATURES):
    frozen = freeze_data(data, season, week, cutoff)
    metadata = team_metadata(frozen.games, season)
    teams = sorted(metadata)
    # Synthetic unscored fixtures expose every team's state at the same instant.
    # They never update form/Elo. Their rest clocks are ignored in neutral comparisons.
    template = frozen.games[frozen.games.season.eq(season)].iloc[0].to_dict()
    fixtures = []
    for i, home in enumerate(teams):
        away = teams[(i + 1) % len(teams)]
        game = {
            **template,
            "game_id": -i - 1,
            "season": season,
            "week": week,
            "start_date": cutoff,
            "home_team": home,
            "away_team": away,
            "home_id": metadata[home]["id"],
            "away_id": metadata[away]["id"],
            "home_conference": metadata[home]["conference"],
            "away_conference": metadata[away]["conference"],
            "home_classification": "fbs",
            "away_classification": "fbs",
            "completed": False,
            "home_points": np.nan,
            "away_points": np.nan,
            "neutral_site": True,
            "conference_game": False,
            "fbs_vs_fbs": True,
        }
        fixtures.append(game)
    # Keep production rest clocks separate from the synthetic fixture replay.
    ordinary = build_point_in_time_features(frozen, include_scheduled=True)
    snapshot_data = replace(
        frozen, games=pd.concat([frozen.games, pd.DataFrame(fixtures)], ignore_index=True)
    )
    snapshot_frame = build_point_in_time_features(snapshot_data, include_scheduled=True)
    snapshots = {}
    for _, row in snapshot_frame[snapshot_frame.game_id.lt(0)].iterrows():
        for side in ("home", "away"):
            snapshots[str(row[f"{side}_team"])] = {
                name.removeprefix(f"{side}_"): float(row[name])
                for name in feature_names
                if name.startswith(f"{side}_")
            }
    assert set(snapshots) == set(teams)
    return frozen, ordinary, snapshots, metadata


def neutral_margin_matrix(estimator, snapshots: dict, week: int, feature_names=CFB_FULL_FEATURES):
    """Predict every neutral equal-rest matchup in both designations."""
    teams = sorted(snapshots)
    n = len(teams)
    home_indices = np.repeat(np.arange(n), n)
    away_indices = np.tile(np.arange(n), n)
    values = {}
    for feature in feature_names:
        if feature.startswith("home_") and feature != "home_field":
            key = feature.removeprefix("home_")
            values[feature] = np.array([snapshots[t].get(key, 7.0) for t in teams])[home_indices]
        elif feature.startswith("away_"):
            key = feature.removeprefix("away_")
            values[feature] = np.array([snapshots[t].get(key, 7.0) for t in teams])[away_indices]
    # Use an equal-rest, neutral, nonconference fixture for every orientation.
    elo = values["home_elo"] - values["away_elo"]
    values.update(
        elo_diff=elo,
        elo_expected_margin=elo / 25.0,
        week=np.full(n * n, week),
        home_field=np.zeros(n * n),
        conference_game=np.zeros(n * n),
        home_rest_days=np.full(n * n, 7.0),
        away_rest_days=np.full(n * n, 7.0),
        rest_advantage=np.zeros(n * n),
    )
    margins = estimator.predict(pd.DataFrame(values)[feature_names]).reshape(n, n)
    return teams, margins


def common_opponent_ratings(
    estimator, snapshots: dict, week: int, feature_names=CFB_FULL_FEATURES
) -> dict[str, float]:
    """Fit the complete symmetric neutral comparison graph, independent of schedule."""
    teams, margins = neutral_margin_matrix(estimator, snapshots, week, feature_names)
    symmetric = (margins - margins.T) / 2.0
    # Complete-graph least squares with mean-zero ratings has this closed form.
    ratings = symmetric.mean(axis=1)
    assert abs(ratings.sum()) < 1e-7
    return dict(zip(teams, ratings.tolist(), strict=True))


def blend_ratings(common, results, weight):
    assert set(common) == set(results)
    return {t: (1 - weight) * common[t] + weight * results[t] for t in common}


def build_blended_cfb_power_ratings(
    data: CFBHistoricalData,
    margin_estimator,
    preseason_estimator,
    feature_names: list[str],
    *,
    created_at: datetime,
    prediction_season: int,
    forecast_week: int,
    model_hash: str,
    input_coverage: dict[str, int],
    display_count: int = 30,
) -> dict[str, Any]:
    """Publish the evaluated 75% common-opponent / 25% results blend."""
    cutoff = pd.Timestamp(created_at)
    frozen, _, snapshots, metadata = frozen_frames(
        data, prediction_season, forecast_week, cutoff, feature_names
    )
    if len(snapshots) < 2:
        raise ValueError("At least two current FBS teams are required for blended rankings")
    first_kickoff = data.games.loc[data.games.season.eq(prediction_season), "start_date"].min()
    _, _, preseason_snapshots, _ = frozen_frames(
        data, prediction_season, 0, first_kickoff, feature_names
    )
    prior = common_opponent_ratings(preseason_estimator, preseason_snapshots, 1, feature_names)
    common = common_opponent_ratings(margin_estimator, snapshots, forecast_week, feature_names)
    season_games = frozen.games[frozen.games.season.eq(prediction_season)]
    results = results_ratings(season_games, common, prior, 4.0, None)
    blended = blend_ratings(common, results, 0.25)
    played = season_games[
        season_games.completed.fillna(False) & season_games.fbs_vs_fbs.fillna(False)
    ]
    scheduled = season_games[
        ~season_games.completed.fillna(False)
        & season_games.fbs_vs_fbs.fillna(False)
        & season_games.start_date.gt(cutoff)
    ]
    ordered = sorted(blended, key=lambda team: (-blended[team], team))
    rows = [
        {
            "rank": rank,
            "team": team,
            "rating": float(blended[team]),
            "common_opponent_rating": float(common[team]),
            "results_rating": float(results[team]),
            "conference": metadata[team]["conference"],
            "completed_games": int((played.home_team.eq(team) | played.away_team.eq(team)).sum()),
            "scheduled_games": int(
                (scheduled.home_team.eq(team) | scheduled.away_team.eq(team)).sum()
            ),
        }
        for rank, team in enumerate(ordered, 1)
    ]
    values = np.array([blended[t] - common[t] for t in ordered])
    pair_errors = (values[:, None] - values[None, :])[np.triu_indices(len(ordered), 1)]
    disagreement_mae = float(np.abs(pair_errors).mean())
    return {
        "schema_version": 1,
        "sport": "college_football",
        "kind": "blended_common_opponent_results",
        "created_at": created_at.isoformat(),
        "prediction_season": int(prediction_season),
        "data_cutoff": created_at.isoformat(),
        "forecast_week": int(forecast_week),
        "preseason_prior_cutoff": first_kickoff.isoformat(),
        "model_hash": model_hash,
        "display_count": min(max(int(display_count), 1), len(rows)),
        "team_count": len(rows),
        "game_count": len(played),
        "neutral_comparison_count": len(rows) * (len(rows) - 1) // 2,
        "method": {
            "common_opponent_weight": 0.75,
            "results_weight": 0.25,
            "preseason_prior_strength": 4.0,
            "margin_cap": None,
        },
        "home_field_points": 3.0,
        "incremental_home_field_points": 3.0,
        "designated_home_bias": 0.0,
        "model_disagreement_mae": disagreement_mae,
        # Preserve readers of the original schema; these now measure disagreement
        # with neutral model margins, not remaining-schedule reconstruction.
        "line_fit_mae": disagreement_mae,
        "line_fit_rmse": float(np.sqrt(np.square(pair_errors).mean())),
        "input_coverage": {key: int(value) for key, value in input_coverage.items()},
        "ratings": rows,
    }


def results_ratings(games, teams, prior, strength, cap, home_edge=3.0):
    teams = sorted(teams)
    index = {team: i for i, team in enumerate(teams)}
    valid = games[
        games.completed.fillna(False)
        & games.fbs_vs_fbs.fillna(False)
        & games.home_team.isin(teams)
        & games.away_team.isin(teams)
    ].dropna(subset=["home_points", "away_points"])
    a = np.zeros((len(valid), len(teams)))
    y = []
    for i, (_, g) in enumerate(valid.iterrows()):
        a[i, index[g.home_team]] = 1.0
        a[i, index[g.away_team]] = -1.0
        margin = float(g.home_points - g.away_points)
        if cap is not None:
            margin = float(np.clip(margin, -cap, cap))
        y.append(margin - (0.0 if bool(g.neutral_site) else home_edge))
    p = np.array([prior[t] for t in teams])
    fitted = np.linalg.solve(
        a.T @ a + strength * np.eye(len(teams)), a.T @ np.asarray(y) + strength * p
    )
    fitted -= fitted.mean()
    return dict(zip(teams, fitted.tolist(), strict=True))
