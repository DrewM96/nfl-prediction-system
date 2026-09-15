from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nfl_prediction.config import is_division_game
from nfl_prediction.data import load_nflverse_data
from nfl_prediction.features import (
    DEFAULT_PRIORS,
    GAME_MARGIN_FEATURES,
    GAME_TOTAL_FEATURES,
    _game_team_summaries,
    _regular_season,
)
from nfl_prediction.historical_market import prequential_component_blend
from nfl_prediction.modeling import GAME_RIDGE_ALPHA, chronological_oof_predictions
from nfl_prediction.roster import attach_roster_transition_features, build_roster_transition_table

VARIANTS = ("baseline", "current_season_2x", "offseason_decay")
SEGMENTS = (
    ("weeks_1_4", 1, 4),
    ("weeks_5_8", 5, 8),
    ("weeks_9_18", 9, 18),
    ("full_season", 1, 18),
)


def _normalized_observation_weights(
    rows: list[dict[str, Any]],
    *,
    current_season: int,
    variant: str,
    recency_lambda: float = 0.85,
    offseason_gamma: float = 0.65,
) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float)
    if variant == "baseline":
        raw = np.ones(len(rows), dtype=float)
    elif variant == "current_season_2x":
        raw = np.asarray(
            [2.0 if int(row["season"]) == current_season else 1.0 for row in rows],
            dtype=float,
        )
    elif variant == "offseason_decay":
        newest_first = list(reversed(rows))
        newest_weights = []
        for games_ago, row in enumerate(newest_first):
            season_gap = max(current_season - int(row["season"]), 0)
            newest_weights.append((recency_lambda**games_ago) * (offseason_gamma**season_gap))
        raw = np.asarray(list(reversed(newest_weights)), dtype=float)
    else:
        raise ValueError(f"Unknown weighting variant: {variant}")
    total = float(raw.sum())
    if total <= 0:
        return np.ones(len(rows), dtype=float)
    # Keep the observed sample's total information mass unchanged so the existing
    # two-game prior shrinkage is identical across variants. Only relative game
    # influence changes.
    return raw * (len(rows) / total)


def _weighted_rolling_shrunk(
    history: list[dict[str, Any]],
    key: str,
    window: int,
    prior: float,
    *,
    current_season: int,
    variant: str,
    prior_weight: float = 2.0,
) -> float:
    observed_rows = [row for row in history[-window:] if pd.notna(row.get(key))]
    if not observed_rows:
        return prior
    weights = _normalized_observation_weights(
        observed_rows,
        current_season=current_season,
        variant=variant,
    )
    values = np.asarray([float(row[key]) for row in observed_rows], dtype=float)
    return float((np.dot(values, weights) + prior * prior_weight) / (weights.sum() + prior_weight))


def _team_state(
    history: list[dict[str, Any]],
    priors: dict[str, float],
    *,
    current_season: int,
    variant: str,
) -> dict[str, float]:
    def roll(key: str, window: int) -> float:
        return _weighted_rolling_shrunk(
            history,
            key,
            window,
            priors[key],
            current_season=current_season,
            variant=variant,
        )

    return {
        "points_for_l4": roll("points_for", 4),
        "points_against_l4": roll("points_against", 4),
        "yards_l4": roll("yards", 4),
        "off_epa_l4": roll("off_epa", 4),
        "def_epa_l4": roll("def_epa", 4),
        "turnovers_l4": roll("turnovers", 4),
        "win_pct_l8": roll("win", 8),
        "pressure_allowed_l4": roll("pressure_allowed", 4),
        "pressure_generated_l4": roll("pressure_generated", 4),
    }


def build_weighted_game_features(
    schedules: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    variant: str,
    rosters: pd.DataFrame | None = None,
    snap_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rebuild only production game features with an experimental history weighting rule."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    schedule = _regular_season(schedules).copy()
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
            if not completed:
                continue
            season = int(game["season"])
            home = str(game["home_team"])
            away = str(game["away_team"])
            home_state = _team_state(histories[home], priors, current_season=season, variant=variant)
            away_state = _team_state(histories[away], priors, current_season=season, variant=variant)
            home_last = histories[home][-1]["gameday"] if histories[home] else None
            away_last = histories[away][-1]["gameday"] if histories[away] else None
            home_rest = min((game_day - home_last).days, 21) if home_last else 7
            away_rest = min((game_day - away_last).days, 21) if away_last else 7
            neutral = str(game.get("location", "Home")).lower() == "neutral"
            home_score = float(game["home_score"])
            away_score = float(game["away_score"])

            row: dict[str, Any] = {
                "game_id": str(game.get("game_id", "")),
                "season": season,
                "week": int(game["week"]),
                "gameday": pd.Timestamp(game["gameday"]),
                "gametime": str(game.get("gametime", "")),
                "home_team": home,
                "away_team": away,
                "neutral_site": neutral,
                "stadium": str(game.get("stadium", "")),
                "home_rest_days": float(home_rest),
                "away_rest_days": float(away_rest),
                "rest_advantage": float(home_rest - away_rest),
                "division_game": float(is_division_game(home, away)),
                "home_field": 0.0 if neutral else 1.0,
                "home_score": home_score,
                "away_score": away_score,
                "total_points": home_score + away_score,
                "home_margin": home_score - away_score,
            }
            row.update({f"home_{key}": value for key, value in home_state.items()})
            row.update({f"away_{key}": value for key, value in away_state.items()})
            rows.append(row)

            game_id = str(game.get("game_id", ""))
            home_summary = summaries.get((game_id, home), {})
            away_summary = summaries.get((game_id, away), {})
            pending_updates.extend(
                [
                    (
                        home,
                        {
                            "season": season,
                            "gameday": game_day,
                            "points_for": home_score,
                            "points_against": away_score,
                            "win": 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0),
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
                            "season": season,
                            "gameday": game_day,
                            "points_for": away_score,
                            "points_against": home_score,
                            "win": 1.0 if away_score > home_score else (0.5 if home_score == away_score else 0.0),
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

    transitions = build_roster_transition_table(rosters, snap_counts)
    return attach_roster_transition_features(pd.DataFrame(rows), transitions)


def _predictions(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    min_train_rows: int,
) -> pd.DataFrame:
    clean = frame.dropna(subset=[*features, target]).copy()
    actual, first, second, indices = chronological_oof_predictions(
        clean,
        features,
        target,
        min_train_rows=min_train_rows,
        ridge_alpha=GAME_RIDGE_ALPHA,
    )
    validation = clean.loc[indices].copy()
    blended, weights = prequential_component_blend(validation, actual, first, second)
    validation["actual"] = actual
    validation["prediction"] = blended
    validation["ridge_weight"] = weights
    return validation


def _metrics(frame: pd.DataFrame, *, target: str) -> dict[str, float | int | None]:
    if frame.empty:
        return {"games": 0, "mae": None, "rmse": None, "bias": None, "winner_accuracy": None}
    actual = frame["actual"].to_numpy(dtype=float)
    pred = frame["prediction"].to_numpy(dtype=float)
    winner_accuracy = None
    if target == "home_margin":
        winner_accuracy = float((np.sign(actual) == np.sign(pred)).mean())
    return {
        "games": int(len(frame)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
        "bias": float(np.mean(pred - actual)),
        "winner_accuracy": winner_accuracy,
    }


def benchmark(
    *,
    seasons: list[int],
    evaluation_seasons: list[int],
    min_train_rows: int,
) -> dict[str, Any]:
    data = load_nflverse_data(seasons)
    report: dict[str, Any] = {
        "seasons": seasons,
        "evaluation_seasons": evaluation_seasons,
        "min_train_rows": min_train_rows,
        "variants": {},
        "weighting": {
            "baseline": "equal observations",
            "current_season_2x": "current-season observations receive 2x relative weight; normalized to preserve prior shrinkage",
            "offseason_decay": "0.85 per-game recency decay and 0.65 per offseason boundary; normalized to preserve prior shrinkage",
        },
    }

    for variant in VARIANTS:
        games = build_weighted_game_features(
            data.schedules,
            data.pbp,
            variant=variant,
            rosters=data.rosters,
            snap_counts=data.snap_counts,
        )
        margin = _predictions(
            games,
            features=GAME_MARGIN_FEATURES,
            target="home_margin",
            min_train_rows=min_train_rows,
        )
        total = _predictions(
            games,
            features=GAME_TOTAL_FEATURES,
            target="total_points",
            min_train_rows=min_train_rows,
        )
        variant_report: dict[str, Any] = {"margin": {}, "total": {}, "by_season": {}}
        for label, start_week, end_week in SEGMENTS:
            margin_slice = margin[
                margin["season"].isin(evaluation_seasons)
                & margin["week"].between(start_week, end_week)
            ]
            total_slice = total[
                total["season"].isin(evaluation_seasons)
                & total["week"].between(start_week, end_week)
            ]
            variant_report["margin"][label] = _metrics(margin_slice, target="home_margin")
            variant_report["total"][label] = _metrics(total_slice, target="total_points")
        for season in evaluation_seasons:
            variant_report["by_season"][str(season)] = {
                "margin": _metrics(margin[margin["season"].eq(season)], target="home_margin"),
                "total": _metrics(total[total["season"].eq(season)], target="total_points"),
            }
        report["variants"][variant] = variant_report
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NFL season-transition weighting benchmark",
        "",
        f"Evaluation seasons: {', '.join(map(str, report['evaluation_seasons']))}",
        "",
        "## Margin MAE",
        "",
        "| Variant | Weeks 1-4 | Weeks 5-8 | Weeks 9-18 | Full season | Winner acc. (full) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        metrics = report["variants"][variant]["margin"]
        values = [metrics[key]["mae"] for key in ("weeks_1_4", "weeks_5_8", "weeks_9_18", "full_season")]
        acc = metrics["full_season"]["winner_accuracy"]
        lines.append(
            f"| {variant} | "
            + " | ".join("—" if value is None else f"{value:.3f}" for value in values)
            + f" | {'—' if acc is None else f'{acc:.1%}'} |"
        )
    lines.extend(
        [
            "",
            "## Total MAE",
            "",
            "| Variant | Weeks 1-4 | Weeks 5-8 | Weeks 9-18 | Full season |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        metrics = report["variants"][variant]["total"]
        values = [metrics[key]["mae"] for key in ("weeks_1_4", "weeks_5_8", "weeks_9_18", "full_season")]
        lines.append(
            f"| {variant} | "
            + " | ".join("—" if value is None else f"{value:.3f}" for value in values)
            + " |"
        )
    lines.extend(["", "## Season-by-season full-season MAE", ""])
    lines.append("| Season | Variant | Margin MAE | Total MAE |")
    lines.append("|---:|---|---:|---:|")
    for season in report["evaluation_seasons"]:
        for variant in VARIANTS:
            season_metrics = report["variants"][variant]["by_season"][str(season)]
            lines.append(
                f"| {season} | {variant} | {season_metrics['margin']['mae']:.3f} | {season_metrics['total']['mae']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--evaluation-seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--min-train-rows", type=int, default=350)
    parser.add_argument("--output-dir", default="reports/season_transition")
    args = parser.parse_args()

    report = benchmark(
        seasons=args.seasons,
        evaluation_seasons=args.evaluation_seasons,
        min_train_rows=args.min_train_rows,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    (output_dir / "benchmark.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
