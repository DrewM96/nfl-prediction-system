from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

from nfl_prediction.pipeline import (
    _add_participation_spines,
    _current_player_snapshots,
    _official_injury_payload,
    _raw_data_fingerprint,
)


def test_snap_spine_preserves_active_zero_target_receiver() -> None:
    keys = ["season", "week", "game_id", "game_date", "posteam", "player_id", "player_name"]
    receiving = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "G",
                "game_date": "2025-09-07",
                "posteam": "A",
                "player_id": "GSIS-1",
                "player_name": "A.One",
                "receiving_yards": 8.0,
                "targets": 1.0,
                "receptions": 1.0,
                "receiving_tds": 0.0,
                "team_targets": 1.0,
            }
        ]
    )
    empty_passing = pd.DataFrame(
        columns=keys + ["passing_yards", "attempts", "completions", "passing_tds", "interceptions"]
    )
    empty_rushing = pd.DataFrame(columns=keys + ["rushing_yards", "carries", "rushing_tds"])
    snaps = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "G",
                "game_type": "REG",
                "pfr_player_id": "P1",
                "position": "WR",
                "team": "A",
                "offense_pct": 0.8,
            },
            {
                "season": 2025,
                "week": 1,
                "game_id": "G",
                "game_type": "REG",
                "pfr_player_id": "P2",
                "position": "WR",
                "team": "A",
                "offense_pct": 0.5,
            },
        ]
    )
    rosters = pd.DataFrame(
        [
            {"season": 2025, "week": 1, "pfr_id": "P1", "gsis_id": "GSIS-1", "full_name": "A One"},
            {"season": 2025, "week": 1, "pfr_id": "P2", "gsis_id": "GSIS-2", "full_name": "B Two"},
        ]
    )
    pbp = pd.DataFrame({"game_id": ["G"], "game_date": ["2025-09-07"]})

    result = _add_participation_spines(
        {"passing": empty_passing, "receiving": receiving, "rushing": empty_rushing},
        snaps,
        rosters,
        pbp,
    )["receiving"]

    zero = result[result["player_id"].eq("GSIS-2")].iloc[0]
    assert zero["targets"] == 0
    assert zero["receptions"] == 0
    assert zero["snap_share"] == 0.5


def test_old_injury_feed_is_marked_stale() -> None:
    injuries = pd.DataFrame([{"season": 2025, "week": 18, "team": "A", "report_status": "Out"}])
    payload = _official_injury_payload(injuries, 2026, datetime(2026, 8, 1, tzinfo=UTC))
    assert payload["stale_for_prediction_season"] is True
    assert payload["available_season"] == 2025


def test_raw_data_fingerprint_changes_with_source_values() -> None:
    source = SimpleNamespace(
        pbp=pd.DataFrame({"game_id": ["G"], "play_id": [1], "epa": [0.1]}),
        schedules=pd.DataFrame({"game_id": ["G"], "home_score": [20]}),
        rosters=pd.DataFrame(),
        injuries=pd.DataFrame(),
        snap_counts=pd.DataFrame(),
    )
    original = _raw_data_fingerprint(source)
    source.pbp.loc[0, "epa"] = 0.2
    assert _raw_data_fingerprint(source) != original


def test_current_player_snapshot_keeps_opponent_epa_and_real_snaps() -> None:
    receiving = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 18,
                "game_id": "G",
                "game_date": "2026-01-04",
                "posteam": "A",
                "player_id": "GSIS-1",
                "player_name": "A.One",
                "receiving_yards": 50.0,
                "targets": 6.0,
                "receptions": 4.0,
                "receiving_tds": 0.0,
                "team_targets": 30.0,
                "snap_share": 0.75,
            }
        ]
    )
    upcoming = pd.DataFrame([{"home_team": "A", "away_team": "B"}])
    roster = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 18,
                "gsis_id": "GSIS-1",
                "team": "A",
                "status": "ACT",
            }
        ]
    )
    result = _current_player_snapshots(
        {"passing": pd.DataFrame(), "receiving": receiving, "rushing": pd.DataFrame()},
        upcoming,
        2026,
        roster,
        {"A": {"def_epa_l4": 0.1}, "B": {"def_epa_l4": -0.2}},
    )["wr"]["GSIS-1"]
    assert result["opponent_def_epa"] == -0.2
    assert result["snap_share_l4"] == 0.75
