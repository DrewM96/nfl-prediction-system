from __future__ import annotations

from typing import Any

from .market import normal_cdf
from .rankings import build_market_power_ratings
from .roster import roster_transition_weight


def _neutral_matchups(predictions: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(game.get("away_team", "")), str(game.get("home_team", "")))
        for game in predictions
        if game.get("neutral_site")
        or not float((game.get("features") or {}).get("home_field", 1.0))
    }


def apply_preseason_calibration(
    predictions: list[dict[str, Any]],
    market_snapshot: dict[str, Any] | None,
    *,
    neutral_matchups: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Blend a preseason market-strength prior into Weeks 1-4 margins.

    The independent football output is retained on every calibrated prediction.
    Market-derived strength starts at full weight in Week 1, then tapers to zero
    after Week 4 as current-season football evidence becomes available.
    """
    snapshot_at = (market_snapshot or {}).get("snapshot_at")
    if predictions and all(
        (game.get("preseason_calibration") or {}).get("snapshot_at") == snapshot_at
        for game in predictions
    ):
        return [dict(game) for game in predictions]
    known_neutral = set(neutral_matchups or set()) | _neutral_matchups(predictions)
    ratings = build_market_power_ratings(
        market_snapshot,
        neutral_matchups=known_neutral,
    )
    if not ratings:
        return [dict(game) for game in predictions]
    rating_by_team = {str(row["team"]): float(row["rating"]) for row in ratings["ratings"]}
    calibrated: list[dict[str, Any]] = []
    for source in predictions:
        game = dict(source)
        home = str(game.get("home_team", ""))
        away = str(game.get("away_team", ""))
        if home not in rating_by_team or away not in rating_by_team:
            calibrated.append(game)
            continue
        football = dict(game.get("football_only") or {})
        football_margin = float(football.get("home_margin", game["predicted_home_margin"]))
        football_total = float(football.get("total", game["total"]))
        football_home_score = float(football.get("home_score", game["home_score"]))
        football_away_score = float(football.get("away_score", game["away_score"]))
        football_probability = float(
            football.get("home_win_probability", game["home_win_probability"])
        )
        football = {
            "home_margin": round(football_margin, 2),
            "total": round(football_total, 1),
            "home_score": round(football_home_score, 1),
            "away_score": round(football_away_score, 1),
            "home_win_probability": round(football_probability, 4),
        }
        home_field = float((game.get("features") or {}).get("home_field", 1.0))
        prior_margin = (
            rating_by_team[home]
            - rating_by_team[away]
            + float(ratings["home_field_points"]) * home_field
        )
        weight = roster_transition_weight(float(game.get("week", 1)))
        margin = weight * prior_margin + (1.0 - weight) * football_margin
        home_score = max((football_total + margin) / 2.0, 0.0)
        away_score = max((football_total - margin) / 2.0, 0.0)
        margin_std = max(float(game.get("margin_std", 1.0)), 1e-6)
        game.update(
            {
                "football_only": football,
                "preseason_calibration": {
                    "snapshot_at": ratings.get("snapshot_at"),
                    "weight": round(weight, 4),
                    "home_rating": round(rating_by_team[home], 4),
                    "away_rating": round(rating_by_team[away], 4),
                    "home_field_points": round(float(ratings["home_field_points"]), 4),
                    "prior_home_margin": round(prior_margin, 2),
                    "neutral_site": not bool(home_field),
                    "method": "decaying consensus-implied team-strength prior",
                },
                "predicted_home_margin": round(margin, 2),
                "spread": round(margin, 2),
                "home_score": round(home_score, 1),
                "away_score": round(away_score, 1),
                "home_win_probability": round(normal_cdf(margin / margin_std), 4),
                "forecast_method": "preseason calibrated" if weight else "football only",
            }
        )
        calibrated.append(game)
    return calibrated
