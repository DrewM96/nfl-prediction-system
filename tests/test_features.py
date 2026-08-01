from __future__ import annotations

import pandas as pd
import pytest

from nfl_prediction.features import (
    add_shifted_rolling_features,
    build_player_game_logs,
    build_point_in_time_game_features,
)


def schedules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_A_B",
                "season": 2025,
                "week": 1,
                "gameday": "2025-09-07",
                "gametime": "13:00",
                "game_type": "REG",
                "home_team": "B",
                "away_team": "A",
                "home_score": 24,
                "away_score": 20,
            },
            {
                "game_id": "2025_02_A_B",
                "season": 2025,
                "week": 2,
                "gameday": "2025-09-14",
                "gametime": "13:00",
                "game_type": "REG",
                "home_team": "A",
                "away_team": "B",
                "home_score": 10,
                "away_score": 40,
            },
            {
                "game_id": "2025_03_A_B",
                "season": 2025,
                "week": 3,
                "gameday": "2025-09-21",
                "gametime": "13:00",
                "game_type": "REG",
                "home_team": "B",
                "away_team": "A",
                "home_score": None,
                "away_score": None,
            },
            {
                "game_id": "2025_01_PRE",
                "season": 2025,
                "week": 1,
                "gameday": "2025-08-10",
                "game_type": "PRE",
                "home_team": "A",
                "away_team": "B",
                "home_score": 99,
                "away_score": 0,
            },
        ]
    )


def pbp() -> pd.DataFrame:
    rows = []
    for game_id, season, week, game_date in [
        ("2025_01_A_B", 2025, 1, "2025-09-07"),
        ("2025_02_A_B", 2025, 2, "2025-09-14"),
    ]:
        for offense, defense in [("A", "B"), ("B", "A")]:
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "game_date": game_date,
                    "season_type": "REG",
                    "posteam": offense,
                    "defteam": defense,
                    "play_type": "pass",
                    "down": 1,
                    "qtr": 1,
                    "score_differential": 0,
                    "yards_gained": 10,
                    "epa": 0.2,
                    "success": 1,
                    "qb_dropback": 1,
                    "qb_hit": 0,
                    "sack": 0,
                    "interception": 0,
                    "fumble_lost": 0,
                    "pass_attempt": 1,
                    "complete_pass": 1,
                    "passing_yards": 10,
                    "pass_touchdown": 0,
                    "passer_player_id": f"QB-{offense}",
                    "passer_player_name": f"Quarterback {offense}",
                    "receiver_player_id": f"WR-{offense}",
                    "receiver_player_name": f"Receiver {offense}",
                    "receiving_yards": 10,
                    "rush_attempt": 0,
                    "qb_kneel": 0,
                    "qb_spike": 0,
                    "rush_touchdown": 0,
                }
            )
    return pd.DataFrame(rows)


def test_future_result_cannot_change_earlier_features() -> None:
    original = build_point_in_time_game_features(schedules(), pbp(), include_unplayed=True).games
    changed_schedule = schedules()
    changed_schedule.loc[changed_schedule["game_id"].eq("2025_02_A_B"), "home_score"] = 70
    changed = build_point_in_time_game_features(
        changed_schedule, pbp(), include_unplayed=True
    ).games
    earlier_columns = [column for column in original.columns if column.endswith(("_l4", "_l8"))]
    pd.testing.assert_series_equal(
        original.loc[original["game_id"].eq("2025_02_A_B"), earlier_columns].iloc[0],
        changed.loc[changed["game_id"].eq("2025_02_A_B"), earlier_columns].iloc[0],
    )


def test_preseason_is_excluded_and_unplayed_game_is_retained() -> None:
    games = build_point_in_time_game_features(schedules(), pbp(), include_unplayed=True).games
    assert "2025_01_PRE" not in set(games["game_id"])
    assert "2025_03_A_B" in set(games["game_id"])


def test_situational_features_use_only_completed_prior_games() -> None:
    original_plays = pbp()
    original = build_point_in_time_game_features(
        schedules(), original_plays, include_unplayed=True
    ).games
    changed_plays = original_plays.copy()
    future = changed_plays["game_id"].eq("2025_02_A_B") & changed_plays["posteam"].eq("A")
    changed_plays.loc[future, ["epa", "success", "yards_gained", "sack"]] = [
        -5.0,
        0,
        99,
        1,
    ]
    changed_plays.loc[future, "passer_player_id"] = "QB-REPLACEMENT"
    changed = build_point_in_time_game_features(
        schedules(), changed_plays, include_unplayed=True
    ).games

    candidate_columns = [
        column
        for column in original.columns
        if any(
            marker in column
            for marker in (
                "success_rate",
                "early_down_epa",
                "explosive_rate",
                "sack_rate",
                "neutral_pass_rate",
                "volatility",
                "qb_continuity",
            )
        )
    ]
    pd.testing.assert_series_equal(
        original.loc[original["game_id"].eq("2025_02_A_B"), candidate_columns].iloc[0],
        changed.loc[changed["game_id"].eq("2025_02_A_B"), candidate_columns].iloc[0],
    )

    week_two = original.loc[original["game_id"].eq("2025_02_A_B")].iloc[0]
    assert week_two["home_success_rate_for_l4"] == 1.0
    assert week_two["home_early_down_epa_for_l4"] == pytest.approx(0.2)
    assert week_two["home_explosive_rate_for_l4"] == 0.0
    assert week_two["home_neutral_pass_rate_l4"] == 1.0
    assert week_two["home_sack_rate_allowed_l4"] == 0.0
    assert week_two["home_qb_continuity_l4"] == 0.5


def test_qb_continuity_is_based_on_prior_primary_passers() -> None:
    original = build_point_in_time_game_features(schedules(), pbp(), include_unplayed=True).games
    changed_plays = pbp()
    changed_plays.loc[
        changed_plays["game_id"].eq("2025_02_A_B") & changed_plays["posteam"].eq("A"),
        "passer_player_id",
    ] = "QB-A2"
    changed = build_point_in_time_game_features(
        schedules(), changed_plays, include_unplayed=True
    ).games

    original_week_three = original.loc[original["game_id"].eq("2025_03_A_B")].iloc[0]
    changed_week_three = changed.loc[changed["game_id"].eq("2025_03_A_B")].iloc[0]
    assert original_week_three["away_qb_continuity_l4"] == 1.0
    assert changed_week_three["away_qb_continuity_l4"] == 0.0


def test_receiving_targets_include_incompletions_and_zero_catch_rows() -> None:
    plays = pbp().iloc[:1].copy()
    incomplete = plays.copy()
    incomplete["receiver_player_id"] = "WR-ZERO"
    incomplete["receiver_player_name"] = "Zero Catch"
    incomplete["complete_pass"] = 0
    incomplete["receiving_yards"] = 0
    incomplete["yards_gained"] = 0
    logs = build_player_game_logs(pd.concat([plays, incomplete], ignore_index=True))["receiving"]
    zero = logs[logs["player_id"].eq("WR-ZERO")].iloc[0]
    assert zero["targets"] == 1
    assert zero["receptions"] == 0
    assert zero["receiving_yards"] == 0


def test_rolling_features_are_shifted() -> None:
    frame = pd.DataFrame(
        {
            "player_id": ["P", "P", "P"],
            "game_id": ["1", "2", "3"],
            "game_date": ["2025-09-01", "2025-09-08", "2025-09-15"],
            "yards": [10.0, 20.0, 1000.0],
        }
    )
    result = add_shifted_rolling_features(frame, ["yards"], windows=(4,))
    assert pd.isna(result.iloc[0]["yards_l4"])
    assert result.iloc[1]["yards_l4"] == 10.0
    assert result.iloc[2]["yards_l4"] < 20.0
