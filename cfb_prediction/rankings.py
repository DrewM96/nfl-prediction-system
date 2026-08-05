from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def build_cfb_power_ratings(
    games: pd.DataFrame,
    predicted_margins: Sequence[float] | np.ndarray,
    *,
    created_at: datetime,
    prediction_season: int,
    data_cutoff: str,
    model_hash: str,
    input_coverage: dict[str, int],
    display_count: int = 30,
    ridge_strength: float = 0.25,
) -> dict[str, Any]:
    """Decompose schedule-wide model margins into neutral-field team ratings.

    Each scheduled matchup contributes one independent model margin. The fitted
    system separates team strength, the designated-home bias at neutral sites,
    and the incremental home-field advantage. A small ridge penalty stabilizes
    conferences connected by relatively few nonconference games.
    """
    required = {"home_team", "away_team", "home_field"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise ValueError(f"CFB ranking games are missing columns: {missing}")
    margins = np.asarray(predicted_margins, dtype=float)
    if len(games) != len(margins):
        raise ValueError("A predicted margin is required for every CFB ranking game")

    valid_games: list[tuple[str, str, float, float]] = []
    for (_, game), margin in zip(games.iterrows(), margins, strict=True):
        home = str(game["home_team"]).strip()
        away = str(game["away_team"]).strip()
        home_field = float(game["home_field"])
        if home and away and home != away and np.isfinite(margin) and np.isfinite(home_field):
            valid_games.append((home, away, home_field, float(margin)))

    teams = sorted({game[0] for game in valid_games} | {game[1] for game in valid_games})
    if len(teams) < 2 or len(valid_games) < len(teams):
        raise ValueError("Insufficient scheduled CFB games to build connected power ratings")

    team_index = {team: index for index, team in enumerate(teams)}
    # Final two columns are designated-home bias and incremental true home field.
    design = np.zeros((len(valid_games), len(teams) + 2), dtype=float)
    targets = np.asarray([game[3] for game in valid_games], dtype=float)
    for row, (home, away, home_field, _) in enumerate(valid_games):
        design[row, team_index[home]] = 1.0
        design[row, team_index[away]] = -1.0
        design[row, -2] = 1.0
        design[row, -1] = home_field

    penalty = np.zeros((len(teams), len(teams) + 2), dtype=float)
    penalty[:, : len(teams)] = np.eye(len(teams)) * np.sqrt(max(ridge_strength, 0.0))
    augmented_design = np.vstack([design, penalty])
    augmented_targets = np.concatenate([targets, np.zeros(len(teams), dtype=float)])
    coefficients = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)[0]
    ratings = coefficients[: len(teams)]
    ratings -= ratings.mean()
    designated_home_bias = float(coefficients[-2])
    home_field_points = float(coefficients[-1])
    fitted = (
        design[:, : len(teams)] @ ratings + designated_home_bias + design[:, -1] * home_field_points
    )
    residuals = fitted - targets
    appearances = {team: sum(team in (game[0], game[1]) for game in valid_games) for team in teams}
    order = np.argsort(ratings)[::-1]
    rows = [
        {
            "rank": rank,
            "team": teams[index],
            "rating": float(ratings[index]),
            "scheduled_games": int(appearances[teams[index]]),
        }
        for rank, index in enumerate(order, start=1)
    ]
    return {
        "schema_version": 1,
        "sport": "college_football",
        "kind": "independent_model_schedule_decomposition",
        "created_at": created_at.isoformat(),
        "prediction_season": int(prediction_season),
        "data_cutoff": data_cutoff,
        "model_hash": model_hash,
        "display_count": min(max(int(display_count), 1), len(rows)),
        "team_count": len(teams),
        "game_count": len(valid_games),
        "home_field_points": designated_home_bias + home_field_points,
        "incremental_home_field_points": home_field_points,
        "designated_home_bias": designated_home_bias,
        "line_fit_mae": float(np.mean(np.abs(residuals))),
        "line_fit_rmse": float(np.sqrt(np.mean(residuals**2))),
        "ridge_strength": float(ridge_strength),
        "input_coverage": {key: int(value) for key, value in input_coverage.items()},
        "ratings": rows,
    }
