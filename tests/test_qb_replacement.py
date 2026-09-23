from __future__ import annotations

import pandas as pd
import pytest

from nfl_prediction.qb_replacement import (
    attach_qb_replacement_features,
    build_qb_replacement_table,
)


def _pbp() -> pd.DataFrame:
    rows = []
    # Prior-season fallback for Week 1.
    for _ in range(20):
        rows.append(
            {
                "season": 2024,
                "week": 18,
                "season_type": "REG",
                "posteam": "A",
                "passer_player_id": "QB1",
                "epa": 0.30,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "sack": 0,
                "success": 1,
            }
        )
    for _ in range(10):
        rows.append(
            {
                "season": 2024,
                "week": 18,
                "season_type": "REG",
                "posteam": "A",
                "passer_player_id": "QB2",
                "epa": -0.10,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "sack": 0,
                "success": 0,
            }
        )

    # Current season Week 1: QB1 clearly remains the usage leader.
    for _ in range(30):
        rows.append(
            {
                "season": 2025,
                "week": 1,
                "season_type": "REG",
                "posteam": "A",
                "passer_player_id": "QB1",
                "epa": 0.40,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "sack": 0,
                "success": 1,
            }
        )
    for _ in range(5):
        rows.append(
            {
                "season": 2025,
                "week": 1,
                "season_type": "REG",
                "posteam": "A",
                "passer_player_id": "QB2",
                "epa": -0.20,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "sack": 0,
                "success": 0,
            }
        )
    # Week 2 rows are deliberately extreme and must not influence Week 2 features.
    for _ in range(40):
        rows.append(
            {
                "season": 2025,
                "week": 2,
                "season_type": "REG",
                "posteam": "A",
                "passer_player_id": "QB2",
                "epa": 2.0,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "sack": 0,
                "success": 1,
            }
        )
    return pd.DataFrame(rows)


def test_qb_replacement_detects_injured_prior_usage_starter() -> None:
    injuries = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "team": "A",
                "position": "QB",
                "gsis_id": "QB1",
                "report_status": "Out",
            }
        ]
    )
    table = build_qb_replacement_table(
        injuries,
        _pbp(),
        shrinkage_dropbacks=0.0,
    )
    row = table.iloc[0]

    assert row["qb_starter_id"] == "QB1"
    assert row["qb_backup_id"] == "QB2"
    assert row["qb_starter_reported"] == 1.0
    assert row["qb_unavailability_weight"] == 1.0
    assert row["qb_starter_epa_per_dropback"] == pytest.approx(0.36)
    assert row["qb_backup_epa_per_dropback"] == pytest.approx(-2.0 / 15.0)
    assert row["qb_expected_dropbacks"] == 35.0
    assert row["qb_expected_points_lost"] == pytest.approx((0.36 + 2.0 / 15.0) * 35.0)


def test_qb_replacement_ignores_nonstarter_qb_injury() -> None:
    injuries = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "team": "A",
                "position": "QB",
                "gsis_id": "QB2",
                "report_status": "Out",
            }
        ]
    )
    table = build_qb_replacement_table(
        injuries,
        _pbp(),
        shrinkage_dropbacks=0.0,
    )
    row = table.iloc[0]

    assert row["qb_starter_id"] == "QB1"
    assert row["qb_starter_reported"] == 0.0
    assert row["qb_unavailability_weight"] == 0.0
    assert row["qb_expected_points_lost"] == 0.0


def test_qb_week_one_uses_prior_season_team_usage() -> None:
    injuries = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "team": "A",
                "position": "QB",
                "gsis_id": "QB1",
                "report_status": "Doubtful",
            }
        ]
    )
    table = build_qb_replacement_table(
        injuries,
        _pbp(),
        shrinkage_dropbacks=0.0,
    )
    row = table.iloc[0]

    assert row["qb_starter_id"] == "QB1"
    assert row["qb_backup_id"] == "QB2"
    assert row["qb_unavailability_weight"] == 0.75
    assert row["qb_expected_dropbacks"] == 30.0
    assert abs(row["qb_expected_points_lost"] - 9.0) < 1e-9


def test_week_one_can_identify_moved_veteran_from_current_roster() -> None:
    pbp_rows = []
    for _ in range(40):
        pbp_rows.append(
            {
                "season": 2024,
                "week": 18,
                "season_type": "REG",
                "posteam": "OLD",
                "passer_player_id": "NEW1",
                "epa": 0.25,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "sack": 0,
                "success": 1,
            }
        )
    for _ in range(8):
        pbp_rows.append(
            {
                "season": 2024,
                "week": 18,
                "season_type": "REG",
                "posteam": "OTHER",
                "passer_player_id": "NEW2",
                "epa": 0.0,
                "qb_dropback": 1,
                "pass_attempt": 1,
                "sack": 0,
                "success": 0,
            }
        )
    injuries = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "team": "A",
                "position": "QB",
                "gsis_id": "NEW1",
                "report_status": "Out",
            }
        ]
    )
    rosters = pd.DataFrame(
        [
            {"season": 2025, "week": 1, "team": "A", "position": "QB", "gsis_id": "NEW1"},
            {"season": 2025, "week": 1, "team": "A", "position": "QB", "gsis_id": "NEW2"},
        ]
    )

    row = build_qb_replacement_table(
        injuries,
        pd.DataFrame(pbp_rows),
        rosters,
        shrinkage_dropbacks=0.0,
    ).iloc[0]

    assert row["qb_starter_id"] == "NEW1"
    assert row["qb_backup_id"] == "NEW2"
    assert row["qb_unavailability_weight"] == 1.0
    assert row["qb_expected_points_lost"] > 0


def test_qb_features_attach_to_correct_game_side() -> None:
    table = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "team": "A",
                "qb_starter_reported": 1.0,
                "qb_unavailability_weight": 1.0,
                "qb_starter_epa_per_dropback": 0.2,
                "qb_backup_epa_per_dropback": 0.0,
                "qb_value_gap_epa_per_dropback": 0.2,
                "qb_expected_dropbacks": 35.0,
                "qb_expected_points_lost": 7.0,
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

    row = attach_qb_replacement_features(games, table).iloc[0]
    assert row["home_qb_expected_points_lost"] == 7.0
    assert row["away_qb_expected_points_lost"] == 0.0
