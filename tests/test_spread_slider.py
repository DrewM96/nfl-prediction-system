"""Exercise the app's HTML helper without loading models or contacting a market feed."""

import ast
import copy
import math
import re
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from nfl_prediction.ui import html_text, spread_label

APP_TREE = ast.parse(Path("app.py").read_text(encoding="utf-8"))


def app_functions(names):
    return "\n\n".join(
        ast.unparse(node)
        for node in APP_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    )


@pytest.fixture
def slider():
    namespace = {"Any": Any, "math": math, "html_text": html_text, "spread_label": spread_label}
    exec(app_functions({"spread_slider_html", "cfb_margin_label"}), namespace)
    return namespace["spread_slider_html"]


@pytest.fixture
def game():
    return {
        "game_id": "chi-phi",
        "home_team": "CHI",
        "away_team": "PHI",
        "predicted_home_margin": 2.0,
        "market_consensus": {"spread": {"home_spread": 3.5}},
    }


@pytest.mark.parametrize("sport", ["nfl", "cfb", "ncaaf"])
@pytest.mark.parametrize("margin,label", [(3.5, "CHI -3.5"), (-3.5, "PHI -3.5"), (0, "Pick")])
def test_favorite_labels_and_pick(slider, game, sport, margin, label):
    game["predicted_home_margin"] = margin
    markup = slider(game, sport, context={"status": "fresh", "home_spread": -margin})
    assert f"GRIDLINE <b>{label}</b>" in markup
    assert f"Market now <b>{label}</b>" in markup
    assert ">even</span>" in markup
    assert "PHI favored" in markup and "CHI favored" in markup


@pytest.mark.parametrize(
    "margin,gap", [(2, "5.5 pts toward CHI"), (-5, "1.5 pts toward PHI"), (-3.48, "even")]
)
def test_gap_direction_and_value(slider, game, margin, gap):
    game["predicted_home_margin"] = margin
    markup = slider(game, "nfl", context={})
    assert f">{gap}</span>" in markup
    assert "Market now <b>PHI -3.5</b>" in markup


def test_track_positions_band_and_movement(slider, game):
    markup = slider(
        game, "nfl", context={"status": "fresh", "home_spread": 3.5, "open_home_spread": 1.0}
    )
    assert 'grid-slider-marker grid-slider-model" style="left:60.00%' in markup
    assert 'grid-slider-marker grid-slider-market" style="left:32.50%' in markup
    assert 'grid-slider-band" style="left:32.50%;width:27.50%' in markup
    assert 'grid-slider-marker grid-slider-open" style="left:45.00%' in markup
    assert 'grid-slider-movement" style="left:32.50%;width:12.50%' in markup
    assert "Open <b>PHI -1.0</b>" in markup


def test_extreme_spreads_are_clamped(slider, game):
    game["predicted_home_margin"] = 10000
    markup = slider(game, "nfl", context={"status": "fresh", "home_spread": 10000})
    assert 'grid-slider-marker grid-slider-model" style="left:98.00%' in markup
    assert 'grid-slider-marker grid-slider-market" style="left:2.00%' in markup
    assert 'grid-slider-band" style="left:2.00%;width:96.00%' in markup


def test_opening_is_included_in_scale_and_zero_is_a_valid_open(slider, game):
    markup = slider(game, "nfl", context={"open_home_spread": 10000})
    assert 'grid-slider-marker grid-slider-open" style="left:2.00%' in markup
    markup = slider(game, "nfl", context={"open_home_spread": 0})
    assert "Open <b>Pick</b>" in markup
    assert 'grid-slider-marker grid-slider-open" style="left:50.00%' in markup


@pytest.mark.parametrize("context", [{}, {"open_home_spread": None}])
def test_missing_opening_has_no_marker_movement_or_legend(slider, game, context):
    markup = slider(game, "nfl", context=context)
    assert "grid-slider-open" not in markup
    assert "grid-slider-movement" not in markup
    assert "Open" not in markup


@pytest.mark.parametrize("status", ["stale", "unavailable"])
@pytest.mark.parametrize("spread", [{"home_spread": 3.5}, {"market_home_margin": -3.5}])
def test_nonfresh_context_falls_back_to_frozen_consensus(slider, game, status, spread):
    game["market_consensus"]["spread"] = spread
    markup = slider(game, "cfb", context={"status": status, "home_spread": -7})
    assert "Market now <b>PHI -3.5</b>" in markup
    assert "CHI -7.0" not in markup


def test_default_context_loads_correct_sport_and_prefers_live(slider, game, monkeypatch):
    loaded = []
    monkeypatch.setitem(
        slider.__globals__, "load_current_market", lambda sport: loaded.append(sport)
    )
    monkeypatch.setitem(
        slider.__globals__, "current_context", lambda *_: {"status": "fresh", "home_spread": -7}
    )
    before = copy.deepcopy(game)
    assert "Market now <b>CHI -7.0</b>" in slider(game, "cfb")
    assert loaded == ["ncaaf"]
    assert game == before


@pytest.mark.parametrize(
    "context",
    [{}, {"status": "stale", "home_spread": -7}, {"status": "fresh", "home_spread": None}],
)
def test_no_market_omits_entire_slider(slider, game, context):
    game.pop("market_consensus")
    assert slider(game, "nfl", context=context) == ""


def test_team_text_is_escaped_everywhere(slider, game):
    game.update(home_team='<img src=x onerror="bad">', away_team="A&B")
    markup = slider(game, "cfb", context={"open_home_spread": 0})
    assert "<img" not in markup
    assert "&lt;img" in markup and "&quot;bad&quot;" in markup and "A&amp;B" in markup


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_hero_and_collapsed_row_render_slider_before_market_panel(sport, tmp_path):
    functions = app_functions(
        {
            "spread_slider_html",
            "cfb_margin_label",
            "cfb_spread_label",
            "cfb_market_spread_label",
            "nfl_market_spread_label",
            "forecast_card_values",
            "render_forecast_header",
            "render_current_market",
            "render_game_row",
            "render_cfb_game_row",
            "render_official_injury_snapshot",
            "_format_injury_snapshot_time",
            "_injury_team_html",
            "_injury_sort_key",
        }
    )
    script = """
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any
import streamlit as st
from nfl_prediction.ui import html_text, spread_label, format_probability, game_matchup_separator
def format_game_time(game): return "Sun 9/13 1p"
format_cfb_game_time = format_game_time
def team_logo_html(*args): return ""
def load_current_market(sport): return {}
def current_context(*args): return {"status": "fresh", "home_spread": 3.5}
def market_context_html(*args): return '<div class="test-market-panel">Current market</div>'
game = dict(game_id="test", home_team="CHI", away_team="PHI", predicted_home_margin=2,
            home_win_probability=.6, total=45, predicted_total=45, home_score=24, away_score=21,
            predicted_home_score=24, predicted_away_score=21,
            market_consensus={"spread": {"home_spread": 3.5, "market_home_margin": -3.5}},
            injury_snapshot={"away": [], "home": [], "available_week": 1, "forecast_week": 1})
"""
    script += functions + f'\nrender_forecast_header(game, "{sport}", featured=True)\n'
    script += f"{'render_game_row' if sport == 'nfl' else 'render_cfb_game_row'}(game, 0)\n"
    script += "render_official_injury_snapshot(game, detailed=True)\n"
    path = tmp_path / "slider_app.py"
    path.write_text(script, encoding="utf-8")
    app = AppTest.from_file(str(path)).run(timeout=15)
    assert not app.exception
    rendered = [m.value for m in app.markdown]
    sliders = [m for m in rendered if 'class="grid-slider"' in m]
    assert len(sliders) == 2
    assert 'class="grid-hero grid-cfb-hero"' in sliders[0]
    assert sliders[0].endswith("</section></div>")
    assert rendered.index(sliders[1]) < next(
        i for i, m in enumerate(rendered) if "test-market-panel" in m
    )
    assert any("Player availability snapshot" in m for m in rendered)
    assert not any("Forecast inputs and availability" in m for m in rendered)
    assert not any(e.label == "Forecast inputs and availability" for e in app.expander)
    assert 'run_every="60s"' in ast.get_source_segment(
        Path("app.py").read_text(encoding="utf-8"),
        next(
            node
            for node in APP_TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "render_current_market"
        ).decorator_list[0],
    )


def test_slider_css_bounds_labels_without_new_breakpoints():
    source = Path("app.py").read_text(encoding="utf-8")
    label_css = re.search(r"\.grid-slider-value \{([^}]+)\}", source)[1]
    assert "clamp(" in label_css and "text-overflow: ellipsis" in label_css
    assert "Forecast inputs and availability" not in source
