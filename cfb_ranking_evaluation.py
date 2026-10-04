"""Chronological research comparison of CFB ranking methods; aggregate output only."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_prediction.client import CFBDClient
from cfb_prediction.data import CFBHistoricalData
from cfb_prediction.features import CFB_FULL_FEATURES, build_point_in_time_features
from cfb_prediction.historical import load_historical_data
from cfb_prediction.modeling import make_ridge
from cfb_prediction.production import _combine
from cfb_prediction.rankings import build_cfb_power_ratings

FEATURES = CFB_FULL_FEATURES
PARAMETERS = [(strength, cap) for strength in (1.0, 4.0, 8.0) for cap in (21.0, 28.0, None)]
BLENDS = {"blend_results25": 0.25, "blend_results50": 0.50, "blend_results75": 0.75}


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


def frozen_frames(data, season, week, cutoff):
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
                for name in FEATURES
                if name.startswith(f"{side}_")
            }
    assert set(snapshots) == set(teams)
    return frozen, ordinary, snapshots, metadata


def neutral_margin_matrix(estimator, snapshots: dict, week: int):
    """Predict every neutral equal-rest matchup in both designations."""
    teams = sorted(snapshots)
    n = len(teams)
    home_indices = np.repeat(np.arange(n), n)
    away_indices = np.tile(np.arange(n), n)
    values = {}
    for feature in FEATURES:
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
    margins = estimator.predict(pd.DataFrame(values)[FEATURES]).reshape(n, n)
    return teams, margins


def common_opponent_ratings(estimator, snapshots: dict, week: int) -> dict[str, float]:
    """Fit the complete symmetric neutral comparison graph, independent of schedule."""
    teams, margins = neutral_margin_matrix(estimator, snapshots, week)
    symmetric = (margins - margins.T) / 2.0
    # Complete-graph least squares with mean-zero ratings has this closed form.
    ratings = symmetric.mean(axis=1)
    assert abs(ratings.sum()) < 1e-7
    return dict(zip(teams, ratings.tolist(), strict=True))


def blend_ratings(common, results, weight):
    assert set(common) == set(results)
    return {t: (1 - weight) * common[t] + weight * results[t] for t in common}


def ranking_example(ratings, metadata, season):
    ordered = sorted(ratings, key=lambda t: (-ratings[t], t))
    return {
        "jmu_rank": ordered.index("James Madison") + 1 if "James Madison" in ordered else None,
        "nonpower_top30": sum(
            not is_power(metadata[t]["conference"], t, season) for t in ordered[:30]
        ),
        "ratings": [
            {"rank": i + 1, "team": t, "rating": ratings[t]} for i, t in enumerate(ordered)
        ],
        "top30": [
            {"rank": i + 1, "team": t, "rating": ratings[t]} for i, t in enumerate(ordered[:30])
        ],
    }


def matchup_audit(ratings, teams, margins):
    """Compare ranking order with an independently queried neutral matchup model."""
    ordered = sorted(ratings, key=lambda t: (-ratings[t], t))[:30]
    indices = {t: i for i, t in enumerate(teams)}
    pairs = []
    for rank, higher in enumerate(ordered, 1):
        for lower_rank, lower in enumerate(ordered[rank:], rank + 1):
            i, j = indices[higher], indices[lower]
            forward = float(margins[i, j])
            reverse = -float(margins[j, i])
            neutral = (forward + reverse) / 2
            pairs.append(
                {
                    "higher_rank": rank,
                    "higher_team": higher,
                    "lower_rank": lower_rank,
                    "lower_team": lower,
                    "rating_margin": ratings[higher] - ratings[lower],
                    "direct_neutral_margin": neutral,
                    "higher_designated_home_margin": forward,
                    "higher_designated_away_margin": reverse,
                    "designation_changes_winner": forward * reverse < 0,
                    "ranking_disagreement": neutral < -1e-8,
                }
            )
    return {
        "pairs": pairs,
        "pair_count": len(pairs),
        "disagreements": sum(p["ranking_disagreement"] for p in pairs),
        "disagreements_over_1_point": sum(p["direct_neutral_margin"] < -1 for p in pairs),
        "disagreements_over_3_points": sum(p["direct_neutral_margin"] < -3 for p in pairs),
        "designation_changes_winner": sum(p["designation_changes_winner"] for p in pairs),
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


def is_power(conference: str, team: str, season: int) -> bool:
    conference = conference.strip().lower()
    return (
        team == "Notre Dame"
        or conference in {"acc", "sec", "big ten", "big 12"}
        or (season <= 2023 and conference in {"pac-12", "pacific-12"})
    )


def metrics(rows, method):
    if not rows:
        return {"games": 0, "mae": None, "rmse": None, "winner_accuracy": None}
    actual = np.array([r["actual"] for r in rows])
    predicted = np.array([r[method] for r in rows])
    errors = predicted - actual
    decisive = actual != 0.0
    return {
        "games": len(rows),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "bias": float(errors.mean()),
        "winner_accuracy": float(((predicted[decisive] > 0) == (actual[decisive] > 0)).mean()),
        "power_side_bias": float(
            np.mean([e * r["power_orientation"] for e, r in zip(errors, rows, strict=True)])
        )
        if all(r["power_orientation"] != 0 for r in rows)
        else None,
    }


def summaries(rows, methods):
    splits = {
        "all": rows,
        "development_2022_2024": [r for r in rows if r["season"] < 2025],
        "holdout_2025": [r for r in rows if r["season"] == 2025],
        "followup_2026": [r for r in rows if r["season"] == 2026],
    }
    return {
        split: {
            group: {method: metrics(subset, method) for method in methods}
            for group, subset in {
                "all": selected,
                "cross_conference": [r for r in selected if r["cross_conference"]],
                "power_vs_nonpower": [r for r in selected if r["power_orientation"] != 0],
            }.items()
        }
        for split, selected in splits.items()
    }


def paired_intervals(rows, method, baseline="schedule_projection", samples=2000):
    groups = {}
    for r in rows:
        key = (r["season"], r["week"])
        groups.setdefault(key, []).append(
            abs(r[method] - r["actual"]) - abs(r[baseline] - r["actual"])
        )
    arrays = [np.array(v) for v in groups.values()]
    if not arrays:
        return None
    rng = np.random.default_rng(20261004)
    draws = []
    for _ in range(samples):
        chosen = rng.integers(0, len(arrays), len(arrays))
        draws.append(float(np.concatenate([arrays[i] for i in chosen]).mean()))
    return {
        "delta_mae": float(np.concatenate(arrays).mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "week_clusters": len(arrays),
    }


def run(output: Path, quick=False):
    client = CFBDClient.from_environment()
    data = _combine(
        [
            load_historical_data(client, list(range(2018, 2026)), max_age=timedelta(days=3650)),
            load_historical_data(client, [2026]),
        ]
    )
    full = build_point_in_time_features(data)
    rows = []
    skipped = []
    current = {}
    current_options = {}
    current_matchups = {}
    candidates = [
        f"results_l{strength:g}_cap{cap if cap is not None else 'none'}"
        for strength, cap in PARAMETERS
    ]
    seasons = (2025, 2026) if quick else range(2022, 2027)
    for season in seasons:
        current_games = data.games[
            data.games.season.eq(season) & data.games.fbs_vs_fbs.fillna(False)
        ]
        first_cutoff = data.games.loc[data.games.season.eq(season), "start_date"].min()
        preseason_train = full[full.season.lt(season)].dropna(subset=[*FEATURES, "home_margin"])
        preseason_model = make_ridge(50.0).fit(
            preseason_train[FEATURES], preseason_train.home_margin
        )
        _, _, preseason_snapshots, _ = frozen_frames(data, season, 0, first_cutoff)
        prior = common_opponent_ratings(preseason_model, preseason_snapshots, 1)
        for week in map(int, sorted(current_games.week.dropna().astype(int).unique())):
            if not 1 <= week <= 11:
                continue
            outcomes = current_games[
                current_games.week.eq(week) & current_games.completed.fillna(False)
            ]
            if outcomes.empty:
                continue
            cutoff = data.games.loc[
                data.games.season.eq(season) & data.games.week.eq(week), "start_date"
            ].min()
            training = full[
                (full.season.lt(season) | (full.season.eq(season) & full.week.lt(week)))
                & full.start_date.lt(cutoff)
            ].dropna(subset=[*FEATURES, "home_margin"])
            assert training.season.max() <= season
            assert not ((training.season == season) & (training.week >= week)).any()
            estimator = make_ridge(50.0).fit(training[FEATURES], training.home_margin)
            frozen, frame, snapshots, metadata = frozen_frames(data, season, week, cutoff)
            future = frame[
                frame.season.eq(season) & frame.start_date.ge(cutoff) & frame.game_id.ge(0)
            ]
            try:
                payload = build_cfb_power_ratings(
                    future,
                    estimator.predict(future[FEATURES]),
                    created_at=cutoff.to_pydatetime(),
                    prediction_season=season,
                    data_cutoff=cutoff.isoformat(),
                    model_hash="research",
                    input_coverage={},
                )
            except ValueError as exc:
                skipped.append({"season": season, "week": week, "reason": str(exc)})
                continue
            schedule = {r["team"]: r["rating"] for r in payload["ratings"]}
            common = common_opponent_ratings(estimator, snapshots, week)
            played = frozen.games[frozen.games.season.eq(season)]
            fitted_results = {
                name: results_ratings(played, common, prior, strength, cap)
                for name, (strength, cap) in zip(candidates, PARAMETERS, strict=True)
            }
            target_features = frame[frame.game_id.isin(outcomes.game_id)].set_index("game_id")
            for _, g in outcomes.iterrows():
                if (
                    g.home_team not in schedule
                    or g.away_team not in schedule
                    or g.game_id not in target_features.index
                ):
                    raise AssertionError(
                        "Comparison is missing an eligible team's rating or frozen game row"
                    )
                power_home = is_power(str(g.home_conference), str(g.home_team), season)
                power_away = is_power(str(g.away_conference), str(g.away_team), season)
                edge = 0.0 if bool(g.neutral_site) else 3.0
                record = {
                    "season": season,
                    "week": week,
                    "actual": float(g.home_points - g.away_points),
                    "cross_conference": str(g.home_conference) != str(g.away_conference),
                    "power_orientation": int(power_home) - int(power_away),
                    "schedule_projection": schedule[g.home_team]
                    - schedule[g.away_team]
                    + float(payload["designated_home_bias"])
                    + (
                        0.0
                        if bool(g.neutral_site)
                        else float(payload["incremental_home_field_points"])
                    ),
                    "common_opponent": common[g.home_team] - common[g.away_team] + edge,
                    "direct_score_model": float(
                        estimator.predict(target_features.loc[[g.game_id], FEATURES])[0]
                    ),
                    **{
                        name: rating[g.home_team] - rating[g.away_team] + edge
                        for name, rating in fitted_results.items()
                    },
                }
                rows.append(record)
            print(
                json.dumps(
                    {
                        "season": season,
                        "week": week,
                        "games": len(outcomes),
                        "training_rows": len(training),
                        "ranking_games": len(future),
                    }
                ),
                flush=True,
            )
        if season == 2026:
            # Evaluate today's rankings without treating their future games as outcomes.
            cutoff = pd.Timestamp(datetime.now(UTC))
            upcoming = current_games[
                ~current_games.completed.fillna(False) & current_games.start_date.gt(cutoff)
            ]
            if not upcoming.empty:
                week = int(upcoming.week.min())
                training = full[
                    (full.season.lt(season) | (full.season.eq(season) & full.week.lt(week)))
                    & full.start_date.lt(cutoff)
                ].dropna(subset=[*FEATURES, "home_margin"])
                estimator = make_ridge(50.0).fit(training[FEATURES], training.home_margin)
                frozen, _, snapshots, metadata = frozen_frames(data, season, week, cutoff)
                common = common_opponent_ratings(estimator, snapshots, week)
                options = {
                    "common_opponent": common,
                    **{
                        name: results_ratings(
                            frozen.games[frozen.games.season.eq(season)],
                            common,
                            prior,
                            strength,
                            cap,
                        )
                        for name, (strength, cap) in zip(candidates, PARAMETERS, strict=True)
                    },
                }
                for name, ratings in options.items():
                    current[name] = ranking_example(ratings, metadata, season)
                current_options = options
                current_teams, current_margins = neutral_margin_matrix(estimator, snapshots, week)
                current_cutoff = cutoff.isoformat()
    development = [r for r in rows if r["season"] < 2025]
    if not development:
        selected = candidates[4]
    else:
        selected = min(candidates, key=lambda name: (metrics(development, name)["mae"], name))
    for row in rows:
        for name, weight in BLENDS.items():
            row[name] = (1 - weight) * row["common_opponent"] + weight * row[selected]
    selected_blend = min(BLENDS, key=lambda name: (metrics(development, name)["mae"], name))
    if current_options:
        for name, weight in BLENDS.items():
            current_options[name] = blend_ratings(
                current_options["common_opponent"], current_options[selected], weight
            )
            current[name] = ranking_example(current_options[name], metadata, 2026)
        for name in ("common_opponent", selected, *BLENDS):
            current_matchups[name] = matchup_audit(
                current_options[name], current_teams, current_margins
            )
    methods = ["schedule_projection", "common_opponent", "direct_score_model", selected, *BLENDS]
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "methodology": {
            "raw_data_published": False,
            "development_seasons": [2022, 2023, 2024],
            "holdout_season": 2025,
            "followup_season": 2026,
            "evaluated_weeks": "1 through 11; late-season underdetermined production fits excluded by design",
            "weekly_freeze": "before the first kickoff of each week; no same-week outcomes in snapshots or model training",
            "common_opponent": "all-team neutral comparisons, both orientations, equal 7-day rest",
            "results_prior": "preseason common-opponent model ratings; fixed 3-point home edge; lambda/cap selected on development MAE only",
            "parameter_grid": PARAMETERS,
            "ridge_alpha": 50.0,
            "model_features": "production elo_form_advanced_preseason",
            "preseason_data_limitation": "returning/talent/recruiting are season-level proxies without historical publication timestamps",
            "historical_schedule_limitation": "final archived schedules are used as schedule-known proxies",
            "holdout_limitation": "2025 has previously been inspected for production model selection; untouched here means no ranking-parameter selection on 2025",
        },
        "selected_results_method": selected,
        "blend_results_weights": BLENDS,
        "selected_blend_on_development": selected_blend,
        "current_snapshot_cutoff": current_cutoff if current_options else None,
        "current_neutral_matchup_audits": current_matchups,
        "candidate_development_metrics": {name: metrics(development, name) for name in candidates},
        "metrics": summaries(rows, methods),
        "by_season": {
            str(s): summaries([r for r in rows if r["season"] == s], methods)["all"]
            for s in seasons
        },
        "holdout_paired_delta_mae_ci95": {
            method: {
                group: paired_intervals(
                    [
                        r
                        for r in rows
                        if r["season"] == 2025
                        and (
                            group == "all"
                            or (group == "cross_conference" and r["cross_conference"])
                            or (group == "power_vs_nonpower" and r["power_orientation"] != 0)
                        )
                    ],
                    method,
                )
                for group in ("all", "cross_conference", "power_vs_nonpower")
            }
            for method in ("common_opponent", selected, *BLENDS)
        },
        "current_ranking_examples": current,
        "skipped_weeks": skipped,
        "comparison_games": len(rows),
        "quick_mode": quick,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected,
                "metrics": result["metrics"],
                "jmu": {name: value["jmu_rank"] for name, value in current.items()},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("reports/cfb-ranking-evaluation/evaluation.json")
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.output, args.quick)
