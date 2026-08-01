from __future__ import annotations

import pandas as pd
import pytest

from nfl_prediction.roster import (
    ROSTER_TEAM_FEATURES,
    attach_roster_transition_features,
    build_roster_transition_table,
    roster_transition_weight,
)


def _snaps() -> pd.DataFrame:
    rows = []
    for team, players in {
        "A": [
            ("QB-A", "QB", 100, 0),
            ("OL-A1", "OL", 100, 0),
            ("OL-A2", "T", 100, 0),
            ("WR-A1", "WR", 100, 0),
            ("WR-A2", "TE", 100, 0),
            ("FRONT-A", "LB", 0, 100),
            ("SECONDARY-A", "CB", 0, 100),
        ],
        "B": [
            ("QB-B", "QB", 100, 0),
            ("OL-B", "OL", 100, 0),
            ("WR-B", "WR", 100, 0),
            ("FRONT-B", "DL", 0, 100),
            ("SECONDARY-B", "S", 0, 100),
        ],
    }.items():
        for player_id, position, offense, defense in players:
            rows.append(
                {
                    "season": 2022,
                    "week": 1,
                    "game_type": "REG",
                    "team": team,
                    "player": player_id,
                    "pfr_player_id": player_id,
                    "position": position,
                    "offense_snaps": offense,
                    "defense_snaps": defense,
                }
            )
    return pd.DataFrame(rows)


def _rosters() -> pd.DataFrame:
    team_players = {
        "A": ["QB-A", "OL-A1", "WR-A1", "FRONT-A", "QB-B"],
        "B": ["OL-B", "WR-B", "SECONDARY-B"],
    }
    return pd.DataFrame(
        [
            {
                "season": 2023,
                "week": 1,
                "game_type": "REG",
                "team": team,
                "pfr_id": player_id,
                "full_name": player_id,
                "status": "ACT",
            }
            for team, player_ids in team_players.items()
            for player_id in player_ids
        ]
    )


def test_roster_transition_uses_prior_snaps_and_player_movement() -> None:
    transitions = build_roster_transition_table(_rosters(), _snaps())
    team_a = transitions[transitions["team"].eq("A")].iloc[0]

    assert team_a["roster_offense_continuity"] == pytest.approx(3 / 5)
    assert team_a["roster_defense_continuity"] == pytest.approx(1 / 2)
    assert team_a["roster_qb_returning"] == 1.0
    assert team_a["roster_ol_continuity"] == pytest.approx(1 / 2)
    assert team_a["roster_skill_continuity"] == pytest.approx(1 / 2)
    assert team_a["roster_front_continuity"] == 1.0
    assert team_a["roster_secondary_continuity"] == 0.0
    assert team_a["roster_incoming_offense_share"] == pytest.approx(1 / 5)


def test_roster_membership_ignores_game_day_inactive_status() -> None:
    active = _rosters()
    inactive = active.copy()
    inactive.loc[inactive["pfr_id"].eq("QB-A"), "status"] = "INA"
    active_result = build_roster_transition_table(active, _snaps())
    inactive_result = build_roster_transition_table(inactive, _snaps())
    pd.testing.assert_frame_equal(active_result, inactive_result)


def test_current_roster_can_match_snap_names_when_pfr_id_is_missing() -> None:
    rosters = _rosters()
    rosters["full_name"] = rosters["pfr_id"]
    rosters.loc[rosters["pfr_id"].eq("OL-A1"), "pfr_id"] = ""
    transitions = build_roster_transition_table(rosters, _snaps())
    team_a = transitions[transitions["team"].eq("A")].iloc[0]
    assert team_a["roster_ol_continuity"] == pytest.approx(1 / 2)


def test_roster_features_taper_to_neutral_after_week_four() -> None:
    transitions = pd.DataFrame(
        [
            {
                "season": 2023,
                "team": team,
                **{feature: value for feature in ROSTER_TEAM_FEATURES},
            }
            for team, value in (("A", 0.2), ("B", -0.2))
        ]
    )
    games = pd.DataFrame(
        [
            {
                "season": 2023,
                "week": week,
                "home_team": "A",
                "away_team": "B",
            }
            for week in (1, 4, 5)
        ]
    )
    result = attach_roster_transition_features(games, transitions)

    assert roster_transition_weight(1) == 1.0
    assert result.iloc[0]["home_roster_offense_continuity_delta"] == pytest.approx(0.2)
    assert result.iloc[1]["home_roster_offense_continuity_delta"] == pytest.approx(0.05)
    assert result.iloc[2]["home_roster_offense_continuity_delta"] == 0.0
