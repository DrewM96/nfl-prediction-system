import pandas as pd
import pytest

from nfl_prediction.features import build_point_in_time_game_features
from nfl_prediction.strength import StrengthConfig, strength_forecasts


def schedule():
    return pd.DataFrame(
        [
            dict(
                game_id="one",
                season=2024,
                week=18,
                gameday="2025-01-01",
                home_team="A",
                away_team="B",
                home_score=40,
                away_score=10,
            ),
            dict(
                game_id="two",
                season=2025,
                week=1,
                gameday="2025-09-01",
                home_team="A",
                away_team="C",
                home_score=30,
                away_score=10,
            ),
            dict(
                game_id="three",
                season=2025,
                week=1,
                gameday="2025-09-03",
                home_team="B",
                away_team="D",
                home_score=20,
                away_score=20,
            ),
            dict(
                game_id="four",
                season=2025,
                week=2,
                gameday="2025-09-08",
                home_team="A",
                away_team="B",
                home_score=25,
                away_score=15,
            ),
        ]
    )


def test_current_week_results_never_change_any_current_week_forecast():
    original = schedule()
    changed = original.copy()
    changed.loc[changed.game_id.eq("two"), "home_score"] = 100
    first = strength_forecasts(original, StrengthConfig())
    second = strength_forecasts(changed, StrengthConfig())
    pd.testing.assert_frame_equal(first[first.week.eq(1)], second[second.week.eq(1)])
    assert first.iloc[-1].strength_margin != second.iloc[-1].strength_margin
    a = build_point_in_time_game_features(original, pd.DataFrame(), freeze_week=True).games
    b = build_point_in_time_game_features(changed, pd.DataFrame(), freeze_week=True).games
    columns = [c for c in a if c.endswith(("_l4", "_l8"))]
    pd.testing.assert_frame_equal(a.loc[a.week.eq(1), columns], b.loc[b.week.eq(1), columns])


def test_offseason_discount_changes_week_one_strength_and_uncertainty():
    weak = strength_forecasts(schedule(), StrengthConfig(carry=0.0))
    strong = strength_forecasts(schedule(), StrengthConfig(carry=1.0))
    assert weak.iloc[1].strength_margin == pytest.approx(2)
    assert strong.iloc[1].strength_margin > weak.iloc[1].strength_margin
    assert strong.iloc[1].strength_uncertainty != weak.iloc[1].strength_uncertainty
