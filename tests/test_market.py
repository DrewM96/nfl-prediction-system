import pytest

from nfl_prediction.market import (
    american_odds_to_implied_probability,
    home_cover_probability,
    sportsbook_home_spread_to_market_margin,
)


def test_sportsbook_favorite_sign_is_converted_once() -> None:
    assert sportsbook_home_spread_to_market_margin(-6.0) == 6.0
    assert home_cover_probability(6.0, -6.0, 10.0) == pytest.approx(0.5)


def test_model_and_market_agreement_is_not_a_fake_edge() -> None:
    probability = home_cover_probability(6.0, -6.0, 10.0)
    assert probability == pytest.approx(0.5)


def test_american_odds_implied_probability() -> None:
    assert american_odds_to_implied_probability(-110) == pytest.approx(110 / 210)
    assert american_odds_to_implied_probability(150) == pytest.approx(100 / 250)
    with pytest.raises(ValueError):
        american_odds_to_implied_probability(0)
