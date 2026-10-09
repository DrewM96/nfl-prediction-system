from __future__ import annotations

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from nfl_prediction.ledger import PredictionLedger
from nfl_prediction.weekly_games import group_weekly_games, result_comparison, weekly_game_key


def game(game_id="one", *, date="2026-10-08"):
    return {
        "game_id": game_id,
        "home_team": "BUF",
        "away_team": "MIA",
        "gameday": date,
        "gametime": "13:00",
        "start_date": f"{date}T17:00:00+00:00",
        "predicted_home_margin": 3.0,
        "total": 43.0,
        "predicted_total": 43.0,
        "home_score": 23.0,
        "away_score": 20.0,
        "predicted_home_score": 23.0,
        "predicted_away_score": 20.0,
        "home_win_probability": 0.6,
        "margin_p10": -10,
        "margin_p90": 15,
        "total_p10": 30,
        "total_p90": 60,
    }


FINAL = {"status": "final", "actual_home_margin": 7, "actual_total": 41}


def test_grouping_keeps_unfinished_order_and_sorts_finals_newest_first():
    games = [game("old", date="2026-10-01"), game("pending"), game("recent"), game("postponed")]
    results = {"old": FINAL, "recent": FINAL, "postponed": {"status": "postponed"}}
    unfinished, completed = group_weekly_games(games, results)
    assert [g["game_id"] for g in unfinished] == ["pending", "postponed"]
    assert [g["game_id"] for g in completed] == ["recent", "old"]
    assert len(unfinished) + len(completed) == len(games)
    assert group_weekly_games([], results) == ([], [])
    assert group_weekly_games(games, {}) == (games, [])


@pytest.mark.parametrize(
    "margin,total,expected",
    [(7, 41, (24, 17)), (0, 0, (0, 0)), (0, 42, (21, 21)), (-7, 41, (17, 24))],
)
def test_comparison_reconstructs_scores_without_changing_forecast(margin, total, expected):
    prediction = game()
    before = prediction.copy()
    comparison = result_comparison(
        prediction, {"actual_home_margin": margin, "actual_total": total}
    )
    assert (comparison["home_score"], comparison["away_score"]) == expected
    assert comparison["margin_error"] == abs(3 - margin)
    assert comparison["total_error"] == abs(43 - total)
    assert prediction == before


def test_missing_comparison_and_stable_identity():
    assert all(value is None for value in result_comparison(game(), {}).values())
    assert all(
        value is None
        for value in result_comparison(
            game(), {"actual_home_margin": float("nan"), "actual_total": "bad"}
        ).values()
    )
    assert weekly_game_key("nfl", "run", game()) != weekly_game_key("cfb", "run", game())
    assert weekly_game_key("nfl", "run", game()) != weekly_game_key("nfl", "other", game())


def test_latest_settlement_correction_is_used(tmp_path):
    ledger = PredictionLedger(tmp_path)
    batch = ledger.record_batch(
        [game()], model_hash="model", data_cutoff="2026-10-01", prediction_season=2026
    )
    ledger.settle(batch.stem, [{"game_id": "one", **FINAL}])
    ledger.settle(batch.stem, [{"game_id": "one", **FINAL, "actual_home_margin": 9}])
    result = ledger.latest_results(batch.stem)["one"]
    assert result_comparison(game(), result)["margin_error"] == 6


def test_legacy_scored_results_are_final_but_explicit_nonfinal_status_wins():
    outcome = {"actual_home_margin": 0, "actual_total": 0}
    assert group_weekly_games([game()], {"one": outcome}) == ([], [game()])
    for status in ("postponed", "cancelled", "scheduled"):
        assert group_weekly_games([game()], {"one": {**outcome, "status": status}}) == (
            [game()],
            [],
        )


def _app_source(sport, all_completed=False, empty=False, no_completed=False):
    # Exercise production page functions with controlled data, without model loading.
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    names = {
        "render_this_week",
        "render_cfb_foundation",
        "render_completed_games",
        "render_game_status",
        "render_forecast_details",
        "cfb_market_spread_label",
        "cfb_margin_label",
    }
    functions = "\n\n".join(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    )
    games = [] if empty else [game("one"), game("two", date="2026-10-09")]
    results = {"one": FINAL, **({"two": FINAL} if all_completed else {})}
    if no_completed:
        results = {}
    state = {
        "status": "data_ready",
        "production_status": "forecast_ready",
        "prediction_season": 2026,
        "schedule": games,
        "prediction_batch": {
            "run_id": "run",
            "predictions": games,
            "metadata": {"forecast_week": 5},
        },
        "model_manifest": {
            "models": {
                k: {"metrics": {"latest_holdout_season": 2025, "latest_holdout_mae": 10}}
                for k in ("margin", "total")
            }
        },
    }
    return f"""
import math
from datetime import datetime
from typing import Any
import streamlit as st
from nfl_prediction.ui import html_text, format_game_time, game_matchup_separator
from nfl_prediction.weekly_games import group_weekly_games, result_comparison, weekly_game_key
{functions}
PREDICTIONS_DIR = CFB_PREDICTIONS_DIR = None
def load_weekly_results(*args): return st.session_state.get("results", {results!r})
def page_header(*args): st.header(args[0])
def published_forecasts(batch): return batch.get("predictions", [])
def render_weekly_picks(*args, **kwargs): pass
def render_cfb_schedule_filters(games, season, completed_ids=None):
    visible = st.session_state.get("visible_ids")
    return games if visible is None else [g for g in games if g["game_id"] in visible]
def _featured_game_score(game): return 1
def render_featured_game(game): st.markdown("Featured: " + game["game_id"])
render_cfb_featured_game = render_featured_game
def render_game_row(game, key): st.markdown("Unfinished: " + game["game_id"])
render_cfb_game_row = render_game_row
format_cfb_game_time = format_game_time
def probability_bar(game): return "Frozen probability"
def nfl_market_edge_label(game): return "Frozen edge"
def nfl_total_label(game): return "43.0"
def game_reasoning(game): return ["Frozen reasoning"]
def market_tile(game): return "Market at forecast"
def render_official_injury_snapshot(*args, **kwargs): st.caption("Frozen injuries")
st.radio("Sport", ["nfl", "cfb"], key="sport", index={0 if sport == "nfl" else 1})
state = {state!r}
(render_this_week if st.session_state.sport == "nfl" else render_cfb_foundation)(state)
""".encode("ascii", errors="backslashreplace").decode("ascii")


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_pages_put_compact_finals_last_and_details_work(sport):
    app = AppTest.from_string(_app_source(sport)).run()
    assert not app.exception
    markdown = [m.value for m in app.markdown]
    assert "Featured: two" in markdown and "Featured: one" not in markdown
    unfinished = markdown.index("Unfinished: two")
    card = next(i for i, value in enumerate(markdown) if "Final score" in value)
    assert unfinished < card
    assert sum("Final score" in value for value in markdown) == 1
    assert any(h.value == "Completed Games (1)" for h in app.subheader)
    assert "BUF 24" in markdown[card] and "MIA 17" in markdown[card]
    assert "4.0 pts" in markdown[card] and "2.0 pts" in markdown[card]
    assert not app.slider
    app.button[0].click().run()
    assert not app.exception
    assert app.button[0].label == "Hide ▲"
    assert any("Frozen probability" in m.value for m in app.markdown)
    # A correction that regroups games does not transfer expansion to another card.
    app.session_state["results"] = {"two": FINAL}
    app.run()
    assert not app.exception
    assert app.button[0].label == "Details ▼"
    app.radio[0].set_value("cfb" if sport == "nfl" else "nfl").run()
    assert not app.exception
    assert app.button[0].label == "Details ▼"


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_all_completed_and_empty_pages(sport):
    app = AppTest.from_string(_app_source(sport, all_completed=True)).run()
    assert not app.exception
    assert any("All games this week are complete" in i.value for i in app.info)
    assert not any("Featured:" in m.value for m in app.markdown)
    cards = [m.value for m in app.markdown if "Final score" in m.value]
    assert len(cards) == 2
    assert "10/9" in cards[0] and "10/8" in cards[1]
    empty = AppTest.from_string(_app_source(sport, empty=True)).run()
    assert not empty.exception
    assert not any(h.value.startswith("Completed Games") for h in empty.subheader)
    unfinished = AppTest.from_string(_app_source(sport, no_completed=True)).run()
    assert not unfinished.exception
    assert not any(h.value.startswith("Completed Games") for h in unfinished.subheader)


def test_college_filters_apply_to_both_sections():
    app = AppTest.from_string(_app_source("cfb")).run()
    app.session_state["visible_ids"] = ["one"]
    app.run()
    assert not app.exception
    assert not any("Featured:" in m.value for m in app.markdown)
    assert any(h.value == "Completed Games (1)" for h in app.subheader)
    app.session_state["visible_ids"] = ["two"]
    app.run()
    assert not app.exception
    assert any("Featured: two" in m.value for m in app.markdown)
    assert not any(h.value.startswith("Completed Games") for h in app.subheader)
    app.session_state["visible_ids"] = []
    app.run()
    assert not app.exception
    assert not any("All games this week are complete" in i.value for i in app.info)
