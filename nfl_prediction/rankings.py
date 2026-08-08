from __future__ import annotations

from typing import Any

import numpy as np


def build_market_power_ratings(
    snapshot: dict[str, Any] | None,
    *,
    neutral_matchups: set[tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Infer zero-sum neutral-field team ratings from consensus spreads.

    The final coefficient estimates the shared home-field edge. Team coefficients
    are relative to an average NFL team and reproduce each observed market home
    margin as ``home rating - away rating + home field``.
    """
    if not snapshot:
        return None
    games: list[dict[str, Any]] = []
    neutral_matchups = neutral_matchups or set()
    single_book_games = 0
    for game in snapshot.get("games", []):
        spread = game.get("spread") or {}
        try:
            margin = float(spread["market_home_margin"])
            book_count = int(spread["book_count"])
        except (KeyError, TypeError, ValueError):
            continue
        home = str(game.get("home_team", "")).strip()
        away = str(game.get("away_team", "")).strip()
        if not home or not away or home == away or not np.isfinite(margin) or book_count <= 0:
            continue
        if book_count == 1:
            single_book_games += 1
            continue
        games.append(
            {
                "home": home,
                "away": away,
                "margin": margin,
                "book_count": book_count,
                "line_iqr": float(spread.get("line_iqr") or 0.0),
                "home_field": float(
                    not (game.get("neutral_site") or (away, home) in neutral_matchups)
                ),
            }
        )
    teams = sorted({game["home"] for game in games} | {game["away"] for game in games})
    if len(teams) < 2 or len(games) < len(teams):
        return None

    team_index = {team: index for index, team in enumerate(teams)}
    design = np.zeros((len(games), len(teams) + 1), dtype=float)
    targets = np.asarray([game["margin"] for game in games], dtype=float)
    weights = np.sqrt(np.asarray([game["book_count"] for game in games], dtype=float))
    for row, game in enumerate(games):
        design[row, team_index[game["home"]]] = 1.0
        design[row, team_index[game["away"]]] = -1.0
        design[row, -1] = game["home_field"]
    weighted_design = design * weights[:, None]
    if np.linalg.matrix_rank(weighted_design) < len(teams):
        return None
    coefficients = np.linalg.lstsq(weighted_design, targets * weights, rcond=None)[0]
    ratings = coefficients[:-1]
    ratings -= ratings.mean()
    home_field = float(coefficients[-1])
    fitted = design[:, :-1] @ ratings + home_field
    residuals = fitted - targets
    appearances = {
        team: sum(team in (game["home"], game["away"]) for game in games) for team in teams
    }
    order = np.argsort(ratings)[::-1]
    rows = [
        {
            "rank": rank,
            "team": teams[index],
            "rating": float(ratings[index]),
            "games": int(appearances[teams[index]]),
        }
        for rank, index in enumerate(order, start=1)
    ]
    return {
        "kind": "market_consensus",
        "snapshot_at": snapshot.get("snapshot_at"),
        "game_count": len(games),
        "excluded_single_book_games": single_book_games,
        "team_count": len(teams),
        "median_book_count": float(np.median([game["book_count"] for game in games])),
        "median_line_iqr": float(np.median([game["line_iqr"] for game in games])),
        "home_field_points": home_field,
        "line_fit_mae": float(np.mean(np.abs(residuals))),
        "line_fit_rmse": float(np.sqrt(np.mean(residuals**2))),
        "ratings": rows,
    }


def build_football_form_ratings(team_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Preserve the existing descriptive recent-form score as a separate view."""
    rows: list[dict[str, Any]] = []
    cutoffs: list[str] = []
    for team, values in team_data.items():
        required = (
            "points_for_l4",
            "points_against_l4",
            "off_epa_l4",
            "def_epa_l4",
            "pressure_generated_l4",
            "pressure_allowed_l4",
        )
        if not all(key in values for key in required):
            continue
        rating = (
            float(values["points_for_l4"])
            - float(values["points_against_l4"])
            + 8.0 * (float(values["off_epa_l4"]) - float(values["def_epa_l4"]))
            + 3.0 * (float(values["pressure_generated_l4"]) - float(values["pressure_allowed_l4"]))
        )
        rows.append({"team": str(team), "rating": rating})
        if values.get("data_cutoff"):
            cutoffs.append(str(values["data_cutoff"]))
    rows.sort(key=lambda row: row["rating"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "kind": "football_form",
        "data_cutoff": max(cutoffs, default=None),
        "team_count": len(rows),
        "ratings": rows,
    }
