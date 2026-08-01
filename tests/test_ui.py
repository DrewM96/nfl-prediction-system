from nfl_prediction.ui import (
    american_moneyline,
    format_american,
    format_game_time,
    game_reasoning,
    market_line_label,
    relative_mae_improvement,
    spread_label,
)


def test_moneyline_and_spread_formatting() -> None:
    assert american_moneyline(0.5) == -100
    assert american_moneyline(0.6) == -150
    assert american_moneyline(0.4) == 150
    assert format_american(150) == "+150"
    assert format_american(-150) == "-150"
    assert (
        spread_label({"home_team": "BUF", "away_team": "MIA", "predicted_home_margin": 3.25})
        == "BUF -3.2"
    )
    assert (
        spread_label({"home_team": "BUF", "away_team": "MIA", "predicted_home_margin": -2.75})
        == "MIA -2.8"
    )


def test_game_time_uses_compact_sportsbook_format() -> None:
    assert format_game_time({"gameday": "2026-09-13", "gametime": "16:25"}) == "Sun 9/13 4:25pm"


def test_reasoning_is_derived_from_real_features() -> None:
    game = {
        "home_team": "BUF",
        "away_team": "MIA",
        "features": {
            "home_off_epa_l4": 0.2,
            "home_def_epa_l4": -0.1,
            "away_off_epa_l4": -0.1,
            "away_def_epa_l4": 0.1,
            "home_points_for_l4": 28.0,
            "home_points_against_l4": 17.0,
            "away_points_for_l4": 17.0,
            "away_points_against_l4": 24.0,
            "home_pressure_generated_l4": 0.25,
            "home_pressure_allowed_l4": 0.1,
            "away_pressure_generated_l4": 0.1,
            "away_pressure_allowed_l4": 0.2,
            "division_game": 1.0,
            "home_field": 1.0,
        },
    }
    reasons = game_reasoning(game)
    assert len(reasons) == 3
    assert all("BUF" in reason for reason in reasons)


def test_reasoning_surfaces_early_season_roster_continuity() -> None:
    reasons = game_reasoning(
        {
            "home_team": "BUF",
            "away_team": "MIA",
            "features": {
                "home_roster_qb_returning_delta": 0.2,
                "home_roster_ol_continuity_delta": 0.2,
                "home_roster_skill_continuity_delta": 0.1,
                "away_roster_qb_returning_delta": -0.2,
                "away_roster_ol_continuity_delta": -0.1,
                "away_roster_skill_continuity_delta": 0.0,
            },
        }
    )
    assert any("BUF returns more continuity" in reason for reason in reasons)


def test_market_tile_uses_consensus_when_available() -> None:
    game = {
        "home_team": "BUF",
        "market_consensus": {
            "provider": "Market consensus",
            "spread": {"home_spread": -3.5},
            "total": {"total": 47.5},
        },
    }
    assert market_line_label(game) == ("Market consensus", "BUF -3.5 · O/U 47.5")
    assert market_line_label({}) == ("Sportsbook line", "— pending feed —")


def test_relative_mae_improvement_is_not_misreported_as_raw_percent() -> None:
    improvement = relative_mae_improvement({"mae": 9.0, "baseline_mae": 10.0})
    assert improvement == 0.1
