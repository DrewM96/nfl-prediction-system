"""Exercise the production ranking fragment with deterministic upcoming quotes."""

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_ranked_cards_book_selection_and_projection_cache(tmp_path):
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    functions = "\n".join(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"cached_prop_projections", "render_prop_rankings"}
    )
    script = """
from datetime import UTC, datetime, timedelta
import pandas as pd
import streamlit as st
from nfl_prediction.current_market import age_seconds
from nfl_prediction.market_ui import age_label
from nfl_prediction.player_props import comparisons, projection_rows
from nfl_prediction.props_ui import prop_card
def team_logo_html(*args): return ""
if "now" not in st.session_state:
    st.session_state.now = datetime.now(UTC)
    st.session_state.inferences = 0
now = st.session_state.now
stamp = now.isoformat()
kickoff = (now + timedelta(days=1)).isoformat()
player = {"player_name": "J.Allen", "team": "BUF", "opponent": "NYJ", "prediction_season": 2026, "roster_season": 2026}
snapshot = {"qb": {"gsis-1": player}, "schedule": [{"game_id": "game", "home_team": "BUF", "away_team": "NYJ", "commence_time": kickoff}], "prediction_season": 2026}
quote = {"game_id": "game", "event_id": "event", "sportsbook": "draftkings", "player_name": "Josh Allen", "player_key": "joshallen", "category": "passing_yards", "team": "BUF", "commence_time": kickoff, "line": 250.5, "over_price": -110, "under_price": -110, "source_timestamp": stamp, "captured_at": stamp}
def load_current_market(key):
    if key == "nfl_depth":
        return {"players": [{"player_id": "gsis-1", "team": "BUF", "player_name": "Josh Allen", "position": "QB", "starter": True, "source_timestamp": stamp}]}
    return {"rows": [quote, {**quote, "sportsbook": "fanduel", "line": 260.5}]}
class Model:
    def distribution(self, frame):
        st.session_state.inferences += 1
        return [{"mean": 270}]
"""
    script += (
        functions
        + """
projections = cached_prop_projections("test-release", snapshot, {"passing_yards": Model()})
render_prop_rankings(projections, "passing_yards")
"""
    )
    path = tmp_path / "prop_ui.py"
    path.write_text(script, encoding="utf-8")
    app = AppTest.from_file(str(path)).run(timeout=30)
    assert not app.exception
    card = next(item.value for item in app.markdown if 'class="prop-board"' in item.value)
    assert "Josh Allen" in card and "+14.5" in card and "255.5" in card
    assert app.session_state.inferences == 1
    app.selectbox[0].set_value("draftkings").run(timeout=30)
    assert not app.exception
    card = next(item.value for item in app.markdown if 'class="prop-board"' in item.value)
    assert "250.5" in card and "+19.5" in card
    assert app.session_state.inferences == 1
