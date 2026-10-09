"""Read-only grouping and comparisons for the displayed weekly forecasts."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from .odds import _game_kickoff


def group_weekly_games(
    games: list[dict[str, Any]], results: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unfinished, completed = [], []
    for game in games:
        result = results.get(str(game["game_id"]), {})
        # Legacy scored batches predate explicit statuses; their recorded outcomes
        # are still authoritative finals. Explicit non-final statuses take precedence.
        final = result.get("status") == "final" or (
            not result.get("status")
            and _number(result.get("actual_home_margin")) is not None
            and _number(result.get("actual_total")) is not None
        )
        (completed if final else unfinished).append(game)
    completed.sort(
        key=lambda game: _game_kickoff(game) or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    return unfinished, completed


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def result_comparison(game: dict[str, Any], result: dict[str, Any]) -> dict[str, float | None]:
    margin, total = _number(result.get("actual_home_margin")), _number(result.get("actual_total"))
    predicted_margin = _number(game.get("predicted_home_margin"))
    predicted_total = _number(game.get("predicted_total", game.get("total")))
    return {
        "home_score": (total + margin) / 2 if total is not None and margin is not None else None,
        "away_score": (total - margin) / 2 if total is not None and margin is not None else None,
        "margin_error": abs(predicted_margin - margin)
        if predicted_margin is not None and margin is not None
        else None,
        "total_error": abs(predicted_total - total)
        if predicted_total is not None and total is not None
        else None,
    }


def weekly_game_key(sport: str, run_id: str, game: dict[str, Any]) -> str:
    return f"{sport}_{run_id}_{game['game_id']}"
