from __future__ import annotations

import pandas as pd

from nfl_prediction.injuries import (
    attach_injury_availability_features,
    build_injury_availability_table,
    filter_to_injury_covered_games,
    injury_unavailability_weight,
)


def _rosters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2025, "week": 1, "gsis_id": "G-QB", "pfr_id": "P-QB"},
            {"season": 2025, "week": 1, "gsis_id": "G-WR", "pfr_id": "P-WR"},
            {"season": 2025, "week": 1, "gsis_id": "G-CB", "pfr_id": "P-CB"},
            {"season": 2024, "week": 1, "gsis_id": "G-QB", "pfr_id": "P-QB"},
        ]
    )


def _snaps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Week 1: team offense/defense each played 60 snaps.
            {
                "season": 2025,
                "week": 1,
                "team": "A",
                "pfr_player_id": "P-QB",
                "position": "QB",
                "offense_snaps": 60,
                "defense_snaps": 0,
                "game_type": "REG",
            },
            {
                "season": 2025,
                "week": 1,
                "team": "A",
                "pfr_player_id": "P-WR",
                "position": "WR",
                "offense_snaps": 30,
                "defense_snaps": 0,
                "game_type": "REG",
            },
            {
                "season": 2025,
                "week": 1,
                "team": "A",
                "pfr_player_id": "P-CB",
                "position": "CB",
                "offense_snaps": 0,
                "defense_snaps": 45,
                "game_type": "REG",
            },
            {
                "season": 2025,
                "week": 1,
                "team": "A",
                "pfr_player_id": "P-OTHER",
                "position": "S",
                "offense_snaps": 0,
                "defense_snaps": 60,
                "game_type": "REG",
            },
            # Week 2 is the game being predicted and must not enter the weight.
            {
                "season": 2025,
                "week": 2,
                "team": "A",
                "pfr_player_id": "P-WR",
                "position": "WR",
                "offense_snaps": 60,
                "defense_snaps": 0,
                "game_type": "REG",
            },
            # Prior-season fallback for Week 1.
            {
                "season": 2024,
                "week": 18,
                "team": "A",
                "pfr_player_id": "P-QB",
                "position": "QB",
                "offense_snaps": 55,
                "defense_snaps": 0,
                "game_type": "REG",
            },
        ]
    )


def test_injury_status_weights_prefer_game_designation() -> None:
    assert (
        injury_unavailability_weight(
            {"report_status": "Out", "practice_status": "Full Participation in Practice"}
        )
        == 1.0
    )
    assert injury_unavailability_weight({"report_status": "Questionable"}) == 0.35
    assert (
        injury_unavailability_weight({"practice_status": "Did Not Participate in Practice"}) == 0.25
    )
    assert (
        injury_unavailability_weight({"practice_status": "Full Participation in Practice"}) == 0.0
    )


def test_injury_availability_uses_only_prior_game_snap_share() -> None:
    injuries = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "team": "A",
                "gsis_id": "G-QB",
                "position": "QB",
                "report_status": "Out",
            },
            {
                "season": 2025,
                "week": 2,
                "team": "A",
                "gsis_id": "G-WR",
                "position": "WR",
                "report_status": "Questionable",
            },
            {
                "season": 2025,
                "week": 2,
                "team": "A",
                "gsis_id": "G-CB",
                "position": "CB",
                "report_status": "Doubtful",
            },
        ]
    )

    table = build_injury_availability_table(injuries, _snaps(), _rosters())
    row = table.iloc[0]

    assert row["injury_out_count"] == 1
    assert row["injury_questionable_count"] == 1
    assert row["injury_doubtful_count"] == 1
    assert row["injury_qb_snap_loss"] == 1.0
    # WR played 30/60 offensive snaps in the only prior current-season game.
    assert row["injury_skill_snap_loss"] == 0.35 * 0.5
    # CB played 45/60 defensive snaps.
    assert row["injury_secondary_snap_loss"] == 0.75 * 0.75


def test_week_one_uses_prior_season_same_team_snap_fallback() -> None:
    injuries = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "team": "A",
                "gsis_id": "G-QB",
                "position": "QB",
                "report_status": "Out",
            }
        ]
    )

    table = build_injury_availability_table(injuries, _snaps(), _rosters())
    assert table.iloc[0]["injury_qb_snap_loss"] == 1.0


def test_injury_features_attach_by_team_and_week() -> None:
    availability = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "team": "A",
                "injury_reported_count": 1.0,
                "injury_out_count": 1.0,
                "injury_doubtful_count": 0.0,
                "injury_questionable_count": 0.0,
                "injury_offense_snap_loss": 1.0,
                "injury_defense_snap_loss": 0.0,
                "injury_qb_snap_loss": 1.0,
                "injury_ol_snap_loss": 0.0,
                "injury_skill_snap_loss": 0.0,
                "injury_front_snap_loss": 0.0,
                "injury_secondary_snap_loss": 0.0,
            }
        ]
    )
    games = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "home_team": "A",
                "away_team": "B",
            }
        ]
    )

    result = attach_injury_availability_features(games, availability).iloc[0]
    assert result["home_injury_qb_snap_loss"] == 1.0
    assert result["away_injury_qb_snap_loss"] == 0.0


def test_games_are_filtered_to_weeks_with_injury_feed() -> None:
    games = pd.DataFrame(
        [
            {"season": 2025, "week": 1, "game_id": "A"},
            {"season": 2025, "week": 2, "game_id": "B"},
        ]
    )
    injuries = pd.DataFrame([{"season": 2025, "week": 2, "team": "A"}])

    result = filter_to_injury_covered_games(games, injuries)
    assert result["game_id"].tolist() == ["B"]
