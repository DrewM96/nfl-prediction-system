from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_prediction.io import atomic_write_json, read_json, sha256_file
from nfl_prediction.results import performance_history, settle_schedule

from .client import CFBDClient
from .config import (
    CFB_HISTORICAL_BENCHMARK_PATH,
    CFB_LATEST_PREDICTION_PATH,
    CFB_MODELS_DIR,
    CFB_PREDICTIONS_DIR,
    PROJECT_ROOT,
)
from .data import CFBHistoricalData
from .features import CFB_FEATURE_CONFIGURATIONS, build_point_in_time_features
from .historical import load_historical_data
from .ledger import record_cfb_prediction_batch
from .modeling import fit_cfb_model, save_cfb_model_bundle
from .rankings import build_cfb_power_ratings


def _combine(parts: list[CFBHistoricalData]) -> CFBHistoricalData:
    def frames(name: str) -> pd.DataFrame:
        values = [getattr(part, name) for part in parts if not getattr(part, name).empty]
        return pd.concat(values, ignore_index=True) if values else pd.DataFrame()

    return CFBHistoricalData(
        games=frames("games"),
        advanced=frames("advanced"),
        returning=frames("returning"),
        portal=frames("portal"),
        talent=frames("talent"),
        recruiting=frames("recruiting"),
        lines=frames("lines"),
    )


def _as_utc(value: date | datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _cut_off_results(data: CFBHistoricalData, as_of: datetime) -> CFBHistoricalData:
    games = data.games.copy()
    after_cutoff = games["start_date"].gt(pd.Timestamp(as_of))
    games.loc[after_cutoff, "completed"] = False
    games.loc[after_cutoff, ["home_points", "away_points"]] = np.nan
    eligible_game_ids = set(games.loc[~after_cutoff, "game_id"].dropna().astype(int))
    advanced = (
        data.advanced[data.advanced["game_id"].isin(eligible_game_ids)].copy()
        if not data.advanced.empty
        else data.advanced
    )
    return replace(data, games=games, advanced=advanced)


def _load_selected_benchmark(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    benchmark = read_json(path)
    if not benchmark or benchmark.get("schema_version") != 1:
        raise ValueError("The fixed schema-v1 CFB benchmark is required")
    selected = benchmark.get("selected_by_development", {})
    if set(selected) != {"margin", "total"}:
        raise ValueError("The benchmark must select margin and total configurations")
    for target, configuration in selected.items():
        if configuration not in CFB_FEATURE_CONFIGURATIONS:
            raise ValueError(f"Unknown {target} feature configuration: {configuration}")
        if (
            benchmark["feature_configurations"].get(configuration)
            != CFB_FEATURE_CONFIGURATIONS[configuration]
        ):
            raise ValueError(f"Benchmark feature schema drift for {configuration}")
    return benchmark, {str(key): str(value) for key, value in selected.items()}


def _current_input_coverage(data: CFBHistoricalData, season: int) -> dict[str, int]:
    games = data.games[data.games["season"].eq(season)]
    teams = set(games.loc[games["fbs_vs_fbs"].fillna(False), "home_team"]) | set(
        games.loc[games["fbs_vs_fbs"].fillna(False), "away_team"]
    )

    def team_count(frame: pd.DataFrame) -> int:
        if not {"season", "team"}.issubset(frame):
            return 0
        return int(frame[frame["season"].eq(season) & frame["team"].isin(teams)]["team"].nunique())

    return {
        "scheduled_fbs_teams": len(teams),
        "returning_production_teams": team_count(data.returning),
        "talent_teams": team_count(data.talent),
        "recruiting_teams": team_count(data.recruiting),
        "portal_records": int(data.portal["season"].eq(season).sum())
        if "season" in data.portal
        else 0,
        "completed_games": int(
            (
                games["completed"].fillna(False)
                & games["home_points"].notna()
                & games["away_points"].notna()
            ).sum()
        ),
    }


def run_cfb_production_update(
    *,
    prediction_season: int,
    historical_seasons: list[int],
    as_of: date | datetime | None = None,
    week: int | None = None,
    refresh_current: bool = False,
    client: CFBDClient | None = None,
    benchmark_path: str | Path = CFB_HISTORICAL_BENCHMARK_PATH,
    models_dir: str | Path = CFB_MODELS_DIR,
    predictions_dir: str | Path = CFB_PREDICTIONS_DIR,
    latest_path: str | Path = CFB_LATEST_PREDICTION_PATH,
    rankings_path: str | Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    timestamp = _as_utc(as_of)
    active_client = client or CFBDClient.from_environment()
    completed_seasons = sorted(
        set(int(value) for value in historical_seasons if value < prediction_season)
    )
    if not completed_seasons:
        raise ValueError("At least one completed historical season is required")
    parts = [
        load_historical_data(
            active_client,
            completed_seasons,
            refresh=False,
            max_age=timedelta(days=3650),
        )
    ]
    parts.append(load_historical_data(active_client, [prediction_season], refresh=refresh_current))
    data = _cut_off_results(_combine(parts), timestamp)
    settle_schedule(predictions_dir, data.games, sport="CFB")
    atomic_write_json(
        Path(latest_path).parent / "performance_history.json", performance_history(predictions_dir)
    )
    features = build_point_in_time_features(data, include_scheduled=True)
    all_scheduled = features[
        features["season"].eq(prediction_season)
        & ~features["completed"].fillna(False)
        & features["start_date"].gt(pd.Timestamp(timestamp))
    ].copy()
    if all_scheduled.empty:
        return {
            "model_hash": None,
            "forecast_week": None,
            "prediction_count": 0,
            "prediction_path": None,
            "rankings_path": None,
            "ranking_count": 0,
            "predictions": [],
            "status": "settled_no_upcoming_games",
        }
    forecast_week = int(week if week is not None else all_scheduled["week"].min())
    # Match the expanding-week validation policy: no current-week outcomes
    # train a model predicting another game in that same week.
    training = features[
        (
            (features["season"] < prediction_season)
            | (features["season"].eq(prediction_season) & features["week"].lt(forecast_week))
        )
        & features["completed"].fillna(False)
        & features["start_date"].lt(pd.Timestamp(timestamp))
    ].copy()
    scheduled = all_scheduled[all_scheduled["week"].eq(forecast_week)].sort_values("start_date")
    if scheduled.empty:
        raise ValueError(f"No upcoming FBS-vs-FBS games are available for Week {forecast_week}")

    benchmark, selected = _load_selected_benchmark(benchmark_path)
    input_coverage = _current_input_coverage(data, prediction_season)
    models = {
        "margin": fit_cfb_model(
            training,
            name="margin",
            feature_names=CFB_FEATURE_CONFIGURATIONS[selected["margin"]],
            target_name="home_margin",
            min_train_rows=int(benchmark["min_train_rows"]),
            alpha=float(benchmark["ridge_alpha"]),
        ),
        "total": fit_cfb_model(
            training,
            name="total",
            feature_names=CFB_FEATURE_CONFIGURATIONS[selected["total"]],
            target_name="total_points",
            min_train_rows=int(benchmark["min_train_rows"]),
            alpha=float(benchmark["ridge_alpha"]),
        ),
    }
    manifest = save_cfb_model_bundle(
        models,
        prediction_season=prediction_season,
        training_seasons=sorted(training["season"].astype(int).unique().tolist()),
        data_cutoff=timestamp.isoformat(),
        selected_configurations=selected,
        input_coverage=input_coverage,
        benchmark_path=benchmark_path,
        git_commit=git_commit,
        output_dir=models_dir,
    )
    manifest_path = Path(models_dir) / "manifest.json"
    model_hash = sha256_file(manifest_path)
    ranking_payload = build_cfb_power_ratings(
        all_scheduled,
        models["margin"].predict(all_scheduled),
        created_at=timestamp,
        prediction_season=prediction_season,
        data_cutoff=timestamp.isoformat(),
        model_hash=model_hash,
        input_coverage=input_coverage,
    )
    ranking_target = (
        Path(rankings_path)
        if rankings_path is not None
        else Path(latest_path).parent / "power_rankings.json"
    )
    atomic_write_json(ranking_target, ranking_payload)
    frozen_rankings = ranking_target.with_name(f"power_rankings-{model_hash}.json")
    atomic_write_json(frozen_rankings, ranking_payload)
    margin_distributions = models["margin"].distribution(scheduled)
    total_distributions = models["total"].distribution(scheduled)
    predictions: list[dict[str, Any]] = []
    for (_, game), margin, total in zip(
        scheduled.iterrows(), margin_distributions, total_distributions, strict=True
    ):
        predicted_home_score = (total["mean"] + margin["mean"]) / 2.0
        predicted_away_score = (total["mean"] - margin["mean"]) / 2.0
        predictions.append(
            {
                "game_id": int(game["game_id"]),
                "season": int(game["season"]),
                "week": int(game["week"]),
                "start_date": pd.Timestamp(game["start_date"]).isoformat(),
                "home_team": str(game["home_team"]),
                "away_team": str(game["away_team"]),
                "neutral_site": not bool(game["home_field"]),
                "predicted_home_margin": float(margin["mean"]),
                "predicted_total": float(total["mean"]),
                "predicted_home_score": float(predicted_home_score),
                "predicted_away_score": float(predicted_away_score),
                "home_win_probability": float(margin["probability_above_zero"]),
                "margin_p10": float(margin["p10"]),
                "margin_p90": float(margin["p90"]),
                "total_p10": float(total["p10"]),
                "total_p90": float(total["p90"]),
                "forecast_type": "independent_football_model",
            }
        )
    target = record_cfb_prediction_batch(
        predictions,
        model_hash=model_hash,
        data_cutoff=timestamp.isoformat(),
        prediction_season=prediction_season,
        metadata={
            "forecast_week": forecast_week,
            "selected_configurations": selected,
            "benchmark_sha256": manifest["benchmark_sha256"],
            "input_coverage": input_coverage,
            "market_data_used": False,
            "provisional": True,
        },
        root=predictions_dir,
        latest_path=latest_path,
        manifest_path=str(
            manifest_path.with_name(f"manifest-{model_hash}.json").relative_to(PROJECT_ROOT)
        ),
        rankings_path=str(frozen_rankings.relative_to(PROJECT_ROOT)),
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "model_hash": model_hash,
        "prediction_path": str(target),
        "rankings_path": str(ranking_target),
        "ranking_count": ranking_payload["display_count"],
        "forecast_week": forecast_week,
        "prediction_count": len(predictions),
        "predictions": predictions,
    }
