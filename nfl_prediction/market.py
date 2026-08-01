from __future__ import annotations

import math


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def sportsbook_home_spread_to_market_margin(home_spread: float) -> float:
    """Convert sportsbook notation (-6 favorite) to expected home margin (+6)."""
    return -float(home_spread)


def american_odds_to_implied_probability(odds: int | float) -> float:
    price = float(odds)
    if price == 0:
        raise ValueError("American odds cannot be zero")
    return (-price) / ((-price) + 100.0) if price < 0 else 100.0 / (price + 100.0)


def home_cover_probability(
    predicted_home_margin: float, sportsbook_home_spread: float, margin_std: float
) -> float:
    market_margin = sportsbook_home_spread_to_market_margin(sportsbook_home_spread)
    return normal_cdf((predicted_home_margin - market_margin) / max(margin_std, 1e-6))


def over_probability(predicted_total: float, market_total: float, total_std: float) -> float:
    return normal_cdf((predicted_total - market_total) / max(total_std, 1e-6))
