from datetime import date

from nfl_prediction.config import get_season_context, is_division_game


def test_preseason_context_loads_upcoming_schedule_season() -> None:
    context = get_season_context(date(2026, 8, 1))
    assert context.prediction_season == 2026
    assert 2025 in context.training_seasons
    assert 2026 in context.training_seasons


def test_context_before_league_year_uses_previous_season() -> None:
    assert get_season_context(date(2026, 2, 1)).prediction_season == 2025


def test_rams_use_nflverse_team_code() -> None:
    assert is_division_game("LA", "SF")
    assert not is_division_game("LA", "KC")
