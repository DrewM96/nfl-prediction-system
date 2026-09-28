from datetime import UTC, datetime

import pandas as pd
import pytest

from nfl_prediction.lineup import attach_lineup_shadow, relative_qb_points, reserve_reports


def test_ongoing_absence_is_not_penalized_twice_and_return_is_positive():
    assert relative_qb_points(-0.1, 0.2, -0.1, 1, 30) == pytest.approx(0)
    assert relative_qb_points(0.2, 0.2, -0.1, 1, 30) == pytest.approx(-9)
    assert relative_qb_points(-0.1, 0.2, -0.1, 0, 30) == pytest.approx(9)


def test_reserve_qb_is_present_without_injury_report():
    rosters = pd.DataFrame(
        [dict(season=2026, week=3, team="A", gsis_id="QB", position="QB", status="IR")]
    )
    reports = reserve_reports(pd.DataFrame(), rosters)
    assert reports.iloc[0].report_status == "Out"


def test_lineup_shadow_is_separate_and_rejects_future_information():
    game = dict(season=2026, week=3, home_team="A", away_team="B", predicted_home_margin=3)
    table = {
        (2026, 3, "A"): dict(eligible=True, expected_points_change=-4),
        (2026, 3, "B"): dict(eligible=True, expected_points_change=0),
    }
    now = datetime(2026, 9, 28, tzinfo=UTC)
    output = attach_lineup_shadow(
        [game], table, captured_at=now.isoformat(), as_of=now, fresh=True
    )[0]
    assert output["predicted_home_margin"] == 3
    assert output["lineup_shadow"]["shadow_margin"] == -1
    assert "lineup_shadow" not in game
    with pytest.raises(ValueError, match="Future"):
        attach_lineup_shadow(
            [game], table, captured_at="2026-09-29T00:00:00Z", as_of=now, fresh=True
        )
