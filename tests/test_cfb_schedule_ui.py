from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cfb_prediction.schedule_ui import filter_schedule, game_conferences, kickoff_day, time_slot

CONFERENCES = {"Alabama": "SEC", "Georgia": "SEC", "Miami": "ACC", "Boise State": "Pac-12"}
GAMES = [
    {
        "game_id": 1,
        "home_team": "Alabama",
        "away_team": "Miami",
        "start_date": "2026-10-03T16:00:00Z",
        "predicted_home_margin": 8,
    },
    {
        "game_id": 2,
        "home_team": "Miami",
        "away_team": "Boise State",
        "start_date": "2026-10-03T19:30:00Z",
        "predicted_home_margin": 3,
    },
    {
        "game_id": 3,
        "home_team": "Boise State",
        "away_team": "Georgia",
        "start_date": "2026-10-04T02:30:00Z",
        "predicted_home_margin": 5,
    },
    {
        "game_id": 4,
        "home_team": "Georgia",
        "away_team": "Alabama",
        "start_date": "2026-10-02T23:00:00Z",
        "predicted_home_margin": 2,
    },
]


@pytest.mark.parametrize(
    "hour,minute,expected",
    [
        (18, 59, "Early"),
        (19, 0, "Afternoon"),
        (22, 59, "Afternoon"),
        (23, 0, "Primetime"),
        (1, 59, "Primetime"),
        (2, 0, "Late"),
    ],
)
def test_slot_boundaries_use_eastern_time(hour, minute, expected):
    day = 4 if hour < 2 or hour == 2 else 3
    assert time_slot({"start_date": f"2026-10-{day:02d}T{hour:02d}:{minute:02d}:00Z"}) == expected


def test_day_and_slot_handle_utc_date_rollover_dst_and_missing_time():
    assert kickoff_day(GAMES[2]) == "Sat 10/3"
    assert time_slot({"start_date": "2026-12-05T20:00:00Z"}) == "Afternoon"
    assert time_slot({"start_date": "2026-10-03T19:00:00"}) == "Afternoon"
    assert kickoff_day({}) == "Time TBD"
    assert time_slot({"start_date": "invalid"}) == "Time TBD"


def test_filters_combine_match_either_team_and_preserve_forecasts():
    before = copy.deepcopy(GAMES)
    assert [g["game_id"] for g in filter_schedule(GAMES, CONFERENCES, conference="SEC")] == [
        1,
        3,
        4,
    ]
    assert [
        g["game_id"]
        for g in filter_schedule(
            GAMES, CONFERENCES, conference="SEC", day="Sat 10/3", slot="Late", search="  GEORGIA "
        )
    ] == [3]
    assert filter_schedule(GAMES, CONFERENCES, conference="SEC", slot="Afternoon") == []
    assert filter_schedule(GAMES, {}) == GAMES
    assert game_conferences({**GAMES[0], "home_conference": "Other"}, CONFERENCES) == {
        "Other",
        "ACC",
    }
    assert before == GAMES


def test_conference_registry_covers_current_forecasts():
    registry = json.loads(Path("data/cfb/team_conferences.json").read_text(encoding="utf-8"))
    assert registry["season"] == 2026
    assert registry["source"].startswith("https://sports.core.api.espn.com/")
    assert len(registry["teams"]) == 138
    assert registry["teams"]["Alabama"] == "SEC"
    assert registry["teams"]["Boise State"] == "Pac-12"


@pytest.fixture
def schedule_app(tmp_path):
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    functions = "\n\n".join(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"reset_cfb_schedule_filters", "render_cfb_schedule_filters", "render_cfb_foundation"}
    )
    script = f"""
from pathlib import Path
from datetime import datetime
from typing import Any
import streamlit as st
from cfb_prediction.schedule_ui import TIME_SLOTS, filter_schedule, game_conferences, kickoff_day, time_slot
from nfl_prediction.ui import html_text
from nfl_prediction.weekly_games import group_weekly_games, weekly_game_key
PROJECT_ROOT = Path('.')
CFB_PREDICTIONS_DIR = PROJECT_ROOT / 'data/cfb/predictions'
def load_weekly_results(*args): return st.session_state.get("results", {{}})
def render_game_status(*args): pass
def render_completed_games(games, *args):
    for game in games: st.markdown(f"completed {{game['game_id']}}")
def read_json(*args): return {{'season': 2026, 'teams': {CONFERENCES!r}}}
def page_header(*args): st.markdown('College Football')
def published_forecasts(*args): return []
def render_weekly_picks(*args, **kwargs): pass
def render_cfb_featured_game(game): st.markdown(f"featured {{game['game_id']}}")
def render_cfb_game_row(game, index): st.markdown(f"card {{game['game_id']}} key {{index}}")
{functions}
metrics = {{'latest_holdout_season': 2025, 'latest_holdout_mae': 8.0}}
state = {{'status': 'data_ready', 'prediction_season': st.session_state.get('season', 2026),
          'prediction_batch': {{'run_id': 'run', 'predictions': {GAMES!r}, 'metadata': {{'forecast_week': 5}}}},
          'model_manifest': {{'models': {{'margin': {{'metrics': metrics}}, 'total': {{'metrics': metrics}}}}}}}}
render_cfb_foundation(state)
"""
    path = tmp_path / "schedule_app.py"
    path.write_text(script, encoding="utf-8")
    return AppTest.from_file(str(path)).run(timeout=15)


def test_filter_controls_update_featured_cards_empty_state_and_reset(schedule_app):
    app = schedule_app
    assert not app.exception
    assert any(
        "Showing <strong>4</strong> of 4 games" in m.value and "Kickoff times ET" in m.value
        for m in app.markdown
    )
    app.get("button_group")[0].set_value(["SEC"]).run()
    app.get("button_group")[1].set_value(["Late"]).run()
    assert not app.exception
    rendered = [m.value for m in app.markdown]
    assert "featured 3" in rendered
    assert "card 3 key cfb_run_3" in rendered
    assert sum(m.startswith("card ") for m in rendered) == 1
    app.text_input(key="cfb_filter_search").set_value("Miami").run()
    assert not app.exception
    assert any("No games match" in message.value for message in app.info)
    assert not any(m.value.startswith(("featured ", "card ")) for m in app.markdown)
    app.button(key="cfb_filter_reset").click().run()
    assert not app.exception
    assert app.session_state["cfb_filter_conference"] == "All conferences"
    assert app.session_state["cfb_filter_slot"] == "All times"
    assert app.text_input(key="cfb_filter_search").value == ""
    assert sum(m.value.startswith("card ") for m in app.markdown) == 4


def test_filter_selection_recovers_when_metadata_season_changes(schedule_app):
    app = schedule_app
    app.get("button_group")[0].set_value(["SEC"]).run()
    app.session_state["season"] = 2027
    app.run()
    assert not app.exception
    assert len(app.get("button_group")[0].options) == 1
    assert app.session_state["cfb_filter_conference"] == "All conferences"
    assert sum(m.value.startswith("card ") for m in app.markdown) == 4


def test_deselecting_quick_pill_restores_all_games(schedule_app):
    app = schedule_app
    app.get("button_group")[0].set_value(["SEC"]).run()
    assert sum(m.value.startswith("card ") for m in app.markdown) == 3
    app.get("button_group")[0].set_value([]).run()
    assert not app.exception
    assert sum(m.value.startswith("card ") for m in app.markdown) == 4


@pytest.mark.parametrize("slot, expected", [("Weekday", [4]), ("Saturday", [1, 2, 3])])
def test_quick_calendar_filters_use_eastern_days(slot, expected):
    assert [g["game_id"] for g in filter_schedule(GAMES, CONFERENCES, slot=slot)] == expected
    extra = [{**GAMES[0], "start_date": "2026-10-04T16:00:00Z"}, {**GAMES[1], "start_date": None}]
    assert filter_schedule(extra, CONFERENCES, slot=slot) == []


def test_completed_filter_uses_recorded_finals_and_combines_with_conference(schedule_app):
    app = schedule_app
    app.session_state["results"] = {
        "1": {"status": "final"},
        "2": {"status": "final"},
        "3": {"status": "in_progress"},
    }
    app.get("button_group")[1].set_value(["Completed"]).run()
    assert not app.exception
    assert [m.value for m in app.markdown if m.value.startswith("completed ")] == [
        "completed 2",
        "completed 1",
    ]
    assert not any(m.value.startswith(("card ", "featured ")) for m in app.markdown)
    app.get("button_group")[0].set_value(["SEC"]).run()
    assert [m.value for m in app.markdown if m.value.startswith("completed ")] == ["completed 1"]
    app.button(key="cfb_filter_reset").click().run()
    assert sum(m.value.startswith("card ") for m in app.markdown) == 2


@pytest.mark.parametrize("slot, count", [("Weekday", 1), ("Saturday", 3), ("Completed", 0)])
def test_new_kickoff_pills_filter_cards_and_empty_state(schedule_app, slot, count):
    app = schedule_app
    app.get("button_group")[1].set_value([slot]).run()
    assert not app.exception
    assert sum(m.value.startswith("card ") for m in app.markdown) == count
    if not count:
        assert any("No games match" in message.value for message in app.info)
