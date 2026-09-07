from datetime import UTC, date, datetime

from cfb_prediction.season import current_cfb_season


def test_cfb_season_stays_with_prior_fall_during_bowl_window() -> None:
    assert current_cfb_season(date(2027, 1, 15)) == 2026
    assert current_cfb_season(datetime(2027, 6, 30, 23, tzinfo=UTC)) == 2026


def test_cfb_season_rolls_over_for_preseason() -> None:
    assert current_cfb_season(date(2027, 7, 1)) == 2027
    assert current_cfb_season(datetime(2027, 12, 1, tzinfo=UTC)) == 2027
