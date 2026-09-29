from datetime import UTC, datetime

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nfl_prediction.prop_results import player_forecast_rows
from nfl_prediction.results import forecast_rows, select_forecasts
from nfl_prediction.results_tracker import (
    grade_pick,
    pick_record,
    score_games,
    team_breakdown,
    weekly_records,
)
from nfl_prediction.results_ui import _game_html, _hero_html, _trend_records


@pytest.mark.parametrize(
    "projection,line,actual,status,winner,expected",
    [
        (5, 3.5, 4, "final", False, "Win"),
        (1, 3.5, 4, "final", False, "Loss"),
        (-1, -3.5, -3, "final", False, "Win"),
        (-6, -3.5, -7, "final", False, "Win"),
        (5, 3, 3, "final", False, "Push"),
        (3.02, 3, 7, "final", False, "No pick"),
        (5, 0, 0, "final", True, "Tie"),
        (0, 0, 7, "final", True, "No pick"),
        (5, None, 7, "final", False, "No line"),
        (None, 3, 7, "final", False, "No projection"),
        (5, 3, None, "final", False, "Pending"),
        (5, 3, 7, "scheduled", False, "Pending"),
        (5, 3, 7, "postponed", False, "Void"),
        (5, 3, 7, "cancelled", False, "Void"),
        (250, 240, 240, "final", False, "Push"),
        (float("nan"), 3, 7, "final", False, "No projection"),
    ],
)
def test_pick_scoring(projection, line, actual, status, winner, expected):
    assert grade_pick(projection, line, actual, status, winner=winner) == expected


def test_records_exclude_nondecisions():
    record = pick_record(
        ["Win", "Win", "Loss", "Push", "Tie", "No line", "No pick", "Pending", "Void"]
    )
    assert record["rate"] == pytest.approx(2 / 3)
    assert record["decisions"] == 3
    assert (
        record["pushes"]
        == record["ties"]
        == record["no_line"]
        == record["no_pick"]
        == record["pending"]
        == record["void"]
        == 1
    )
    assert pick_record(["Push", "No line"])["rate"] is None


def _game(**kwargs):
    return dict(
        {
            "game_id": "one",
            "season": 2026,
            "week": 1,
            "home_team": "CHI",
            "away_team": "PHI",
            "kickoff": datetime(2026, 9, 13, tzinfo=UTC),
            "status": "final",
            "published_margin": 6,
            "independent_margin": -2,
            "published_total": 46,
            "independent_total": 42,
            "market_margin": 3,
            "market_total": 44,
            "actual_margin": 10,
            "actual_total": 50,
        },
        **kwargs,
    )


def test_source_changes_picks_and_preserves_input():
    rows = pd.DataFrame([_game()])
    before = rows.copy(deep=True)
    published = score_games(rows).iloc[0]
    independent = score_games(rows, source="independent").iloc[0]
    assert (published.winner_result, published.ats_result, published.total_result) == (
        "Win",
        "Win",
        "Win",
    )
    assert (independent.winner_result, independent.ats_result, independent.total_result) == (
        "Loss",
        "Loss",
        "Loss",
    )
    pd.testing.assert_frame_equal(rows, before)


def test_team_score_accuracy_and_backed_side_are_distinct():
    scored = score_games(pd.DataFrame([_game()]))
    teams = team_breakdown(scored).set_index("team")
    # CHI projected 26, actual 30; PHI projected 20, actual 20.
    assert teams.loc["CHI", "score_mae"] == 4
    assert teams.loc["PHI", "score_mae"] == 0
    assert teams.loc["CHI", "score_bias"] == -4
    assert teams.loc["CHI", "ats_wins"] == 1
    assert teams.loc["PHI", "ats_picks"] == 0
    assert teams.winner_wins.tolist() == [1, 1]
    assert team_breakdown(scored, minimum=2).empty
    assert pick_record(scored.winner_result)["wins"] == 1


def test_away_ats_backing_and_pending_games():
    scored = score_games(
        pd.DataFrame(
            [
                _game(published_margin=-6, actual_margin=-10),
                _game(game_id="two", status="scheduled"),
            ]
        )
    )
    teams = team_breakdown(scored).set_index("team")
    assert teams.loc["PHI", "ats_wins"] == 1
    assert teams.loc["CHI", "ats_picks"] == 0
    assert teams.games.tolist() == [1, 1]
    assert pd.isna(scored.iloc[1].margin_error)


def test_weekly_cumulative_records_exclude_pending():
    rows = score_games(
        pd.DataFrame(
            [
                _game(),
                _game(game_id="two", week=2, actual_margin=-10),
                _game(game_id="three", week=3, status="scheduled"),
            ]
        )
    )
    weekly = weekly_records(rows)
    cumulative = weekly_records(rows, cumulative=True)
    assert set(weekly.week) == {1, 2}
    assert weekly.query("week == 2 and series == 'Winners'").iloc[0].rate == 0
    assert cumulative.query("week == 2 and series == 'Winners'").iloc[0].rate == 0.5


def test_matched_error_chart_uses_identical_samples():
    rows = pd.DataFrame([_game(), _game(game_id="two", market_margin=None, actual_margin=100)])
    trend = _trend_records(rows, target="margin", cumulative=False)
    assert [r["games"] for r in trend] == [1, 1, 1]
    assert [r["mae"] for r in trend] == [4, 12, 7]


def test_results_html_escapes_team_names_and_uses_correct_spread_sign():
    rows = score_games(pd.DataFrame([_game(home_team='<img onerror="bad">', market_margin=-3.5)]))
    markup = _game_html(rows.iloc[0].to_dict(), source="published")
    assert "<img" not in markup and "&lt;img" in markup
    assert "+3.5" in markup
    assert "PHI 20" in markup
    assert "is-win" in markup
    hero = _hero_html(rows, league="<script>", season=2026, through=1, source="published")
    assert "<script>" not in hero and "&lt;script&gt;" in hero


@pytest.mark.parametrize(
    "league,root", [("NFL", "data/predictions"), ("CFB", "data/cfb/predictions")]
)
def test_results_filters_and_source_switch(league, root):
    script = f"from nfl_prediction.results_ui import render_results\nrender_results({root!r}, league={league!r})"
    app = AppTest.from_string(script).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Overview", "Wagon tracker", "Game log"] + (
        ["Player props"] if league == "NFL" else []
    )
    assert any("grid-results-hero" in m.value for m in app.markdown)
    expected = pick_record(
        score_games(select_forecasts(forecast_rows(root), policy="horizon")).ats_result
    )
    assert any(
        f"{expected['wins']}–{expected['losses']}–{expected['pushes']}" in m.value
        for m in app.markdown
    )
    next(s for s in app.selectbox if s.label == "Forecast source").select("Independent").run(
        timeout=20
    )
    assert not app.exception
    next(s for s in app.selectbox if s.label == "Show").select("ATS losses").run(timeout=20)
    assert not app.exception
    next(s for s in app.selectbox if s.label == "Forecast selection").select("First published").run(
        timeout=20
    )
    assert not app.exception
    if league == "NFL":
        # Weekly updates start archiving player projections after the legacy batches.
        no_player_history = player_forecast_rows(root, policy="first").empty
        assert (
            any("Earlier releases did not archive" in m.value for m in app.markdown)
            == no_player_history
        )
