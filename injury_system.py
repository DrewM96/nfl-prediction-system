"""Session-scoped manual injury scenarios.

Official injury data belongs in the weekly data pipeline. This module lets a
user explore an explicit scenario without mutating shared production files.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from nfl_prediction.io import atomic_write_json, read_json

AVAILABILITY_PROBABILITY = {
    "OUT": 0.0,
    "DOUBTFUL": 0.25,
    "QUESTIONABLE": 0.65,
}

# Estimated points lost when unavailable. These are transparent scenario
# assumptions, not trained model features or claims of causal value.
OFFENSIVE_POINT_VALUE = {
    "QB": 4.5,
    "RB1": 0.7,
    "RB2": 0.2,
    "WR1": 0.9,
    "WR2": 0.4,
    "TE1": 0.5,
    "OL_MULTIPLE": 1.0,
}
DEFENSIVE_POINT_VALUE = {"DEF_STAR": 0.8}


class InjuryAdjustmentSystem:
    def __init__(self, injury_file: str | Path = "injuries.json", *, persist: bool = True):
        self.injury_file = Path(injury_file)
        self.persist = persist
        if persist:
            self.injuries = read_json(
                self.injury_file,
                {"last_updated": datetime.now(UTC).isoformat(), "injuries": []},
            )
        else:
            self.injuries = st.session_state.setdefault(
                "injury_scenario",
                {"last_updated": datetime.now(UTC).isoformat(), "injuries": []},
            )

    def _save_injuries(self) -> None:
        self.injuries["last_updated"] = datetime.now(UTC).isoformat()
        if self.persist:
            atomic_write_json(self.injury_file, self.injuries)
        else:
            st.session_state["injury_scenario"] = self.injuries

    def add_injury(
        self, team: str, player_name: str, position: str, status: str, notes: str = ""
    ) -> None:
        self.injuries["injuries"] = [
            injury
            for injury in self.injuries.get("injuries", [])
            if not (injury["team"] == team and injury["player_name"] == player_name)
        ]
        self.injuries["injuries"].append(
            {
                "team": team,
                "player_name": player_name,
                "position": position,
                "status": status,
                "notes": notes,
                "added_at": datetime.now(UTC).isoformat(),
            }
        )
        self._save_injuries()

    def remove_injury(self, team: str, player_name: str) -> None:
        self.injuries["injuries"] = [
            injury
            for injury in self.injuries.get("injuries", [])
            if not (injury["team"] == team and injury["player_name"] == player_name)
        ]
        self._save_injuries()

    def clear(self) -> None:
        self.injuries["injuries"] = []
        self._save_injuries()

    def get_team_injuries(self, team: str) -> list[dict[str, Any]]:
        return [injury for injury in self.injuries.get("injuries", []) if injury["team"] == team]

    def expected_score_effect(self, team: str) -> tuple[float, float, list[str]]:
        offense_points_lost = 0.0
        opponent_points_added = 0.0
        notes = []
        for injury in self.get_team_injuries(team):
            availability = AVAILABILITY_PROBABILITY.get(injury["status"], 1.0)
            unavailable = 1.0 - availability
            position = injury["position"]
            if position in OFFENSIVE_POINT_VALUE:
                value = OFFENSIVE_POINT_VALUE[position] * unavailable
                offense_points_lost += value
            elif position in DEFENSIVE_POINT_VALUE:
                value = DEFENSIVE_POINT_VALUE[position] * unavailable
                opponent_points_added += value
            else:
                value = 0.0
            notes.append(
                f"{injury['player_name']} ({position}, {injury['status']}): "
                f"{availability:.0%} availability assumption, {value:.1f}-point scenario effect"
            )
        return offense_points_lost, opponent_points_added, notes

    def adjust_game_prediction(
        self, prediction: dict[str, Any], home_team: str, away_team: str
    ) -> dict[str, Any]:
        home_offense_lost, home_opponent_added, home_notes = self.expected_score_effect(home_team)
        away_offense_lost, away_opponent_added, away_notes = self.expected_score_effect(away_team)
        home_score = max(
            float(prediction["home_score"]) - home_offense_lost + away_opponent_added, 0.0
        )
        away_score = max(
            float(prediction["away_score"]) - away_offense_lost + home_opponent_added, 0.0
        )
        home_margin = home_score - away_score
        adjusted = dict(prediction)
        adjusted.update(
            {
                "home_score": round(home_score, 1),
                "away_score": round(away_score, 1),
                "predicted_home_margin": round(home_margin, 1),
                "spread": round(home_margin, 1),
                "total": round(home_score + away_score, 1),
                "injury_adjusted": bool(home_notes or away_notes),
                "adjustment_note": "; ".join(home_notes + away_notes),
            }
        )
        return adjusted

    def adjust_player_prediction(
        self, prediction: float, player_name: str, team: str
    ) -> tuple[float, str]:
        injury = next(
            (
                item
                for item in self.get_team_injuries(team)
                if item["player_name"].casefold() == player_name.casefold()
            ),
            None,
        )
        if not injury:
            return prediction, "Healthy"
        availability = AVAILABILITY_PROBABILITY.get(injury["status"], 1.0)
        return (
            round(max(prediction * availability, 0.0), 1),
            f"Manual {injury['status']} scenario: {availability:.0%} availability assumption",
        )


def render_injury_manager(
    injury_system: InjuryAdjustmentSystem, available_teams: list[str]
) -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Injury scenario")
    st.sidebar.caption(
        "Session-only manual assumptions; official reports come from the update pipeline."
    )
    injuries = injury_system.injuries.get("injuries", [])
    if injuries:
        with st.sidebar.expander(f"Current scenario ({len(injuries)})"):
            for injury in injuries:
                st.write(
                    f"{injury['team']} · {injury['player_name']} · "
                    f"{injury['position']} · {injury['status']}"
                )
    with st.sidebar.expander("Add or update"):
        team = st.selectbox("Team", available_teams, key="injury_team")
        player = st.text_input("Player", key="injury_player")
        position = st.selectbox(
            "Position",
            ["QB", "RB1", "RB2", "WR1", "WR2", "TE1", "OL_MULTIPLE", "DEF_STAR"],
            key="injury_position",
        )
        status = st.selectbox("Status", ["OUT", "DOUBTFUL", "QUESTIONABLE"], key="injury_status")
        notes = st.text_input("Notes", key="injury_notes")
        if st.button("Apply scenario") and player.strip():
            injury_system.add_injury(team, player.strip(), position, status, notes)
            st.rerun()
        if st.button("Clear scenario"):
            injury_system.clear()
            st.rerun()
    if injuries:
        with st.sidebar.expander("Remove"):
            choices = [(item["team"], item["player_name"]) for item in injuries]
            selected = st.selectbox(
                "Player", choices, format_func=lambda value: f"{value[0]} · {value[1]}"
            )
            if st.button("Remove selected"):
                injury_system.remove_injury(*selected)
                st.rerun()


def integrate_injuries_into_game_prediction(
    base_prediction: dict[str, Any],
    injury_system: InjuryAdjustmentSystem,
    team1: str,
    team2: str,
    home_team: str,
) -> dict[str, Any]:
    away_team = team2 if home_team == team1 else team1
    home_score = float(
        base_prediction.get(
            "home_score",
            base_prediction["team1_score"]
            if team1 == home_team
            else base_prediction["team2_score"],
        )
    )
    away_score = float(
        base_prediction.get(
            "away_score",
            base_prediction["team2_score"]
            if team2 == away_team
            else base_prediction["team1_score"],
        )
    )
    adjusted = injury_system.adjust_game_prediction(
        {**base_prediction, "home_score": home_score, "away_score": away_score},
        home_team,
        away_team,
    )
    adjusted["team1_score"] = (
        adjusted["home_score"] if team1 == home_team else adjusted["away_score"]
    )
    adjusted["team2_score"] = (
        adjusted["home_score"] if team2 == home_team else adjusted["away_score"]
    )
    return adjusted


def integrate_injuries_into_player_prediction(
    prediction_value: float,
    injury_system: InjuryAdjustmentSystem,
    player_name: str,
    team: str,
    prop_type: str | None = None,
) -> tuple[float, str]:
    del prop_type
    return injury_system.adjust_player_prediction(prediction_value, player_name, team)
