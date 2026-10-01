import copy
import json
import sqlite3
from datetime import timedelta

import pandas as pd
import pytest
import test_market_deployment
import test_owls
from streamlit.testing.v1 import AppTest
from test_owls import NOW, STAMP, Client

from nfl_prediction.current_market import MarketStore, poll_market
from nfl_prediction.money_signals import (
    first_signal_records,
    published_forecasts,
    qualifying_signals,
)
from nfl_prediction.owls import parse_odds, parse_splits

odds = test_owls.odds
slate = test_owls.slate
splits = test_owls.splits
pg_store = test_market_deployment.pg_store


@pytest.fixture
def inputs(odds, splits, slate):
    forecast = {**slate[0], "predicted_home_margin": 6, "total": 50, "season": 2026, "week": 2}
    for source in splits["data"][0]["splits"]:
        for kind, sides in (("spread", ("home", "away")), ("total", ("over", "under"))):
            source[kind] = {
                f"{side}_{metric}_pct": value
                for metric, values in (("bets", (40, 60)), ("handle", (65, 35)))
                for side, value in zip(sides, values, strict=True)
            }
    board = parse_odds(odds, "nfl", [forecast], STAMP)
    board["games"][0]["splits"] = parse_splits(splits)[test_owls.EVENT]
    return forecast, board, odds, splits


@pytest.mark.parametrize("sport", ["nfl", "ncaaf"])
def test_both_markets_preserve_complete_inputs_and_projection(inputs, sport):
    forecast, board, _, _ = inputs
    board["games"][0]["sport"] = sport
    if sport == "ncaaf":
        forecast["predicted_total"] = forecast.pop("total")
    before = copy.deepcopy((forecast, board))
    rows = qualifying_signals(forecast, board, now=NOW)
    assert [r["market"] for r in rows] == ["spread", "total"]
    assert [r["handle_side"] for r in rows] == ["home", "over"]
    assert [r["line"] for r in rows] == [-3.75, 45.5]
    assert all(r["ticket_pct"] == 60 and r["handle_pct"] == 65 for r in rows)
    assert all(len(r["split_books"]) == len(r["odds_books"]) == 2 for r in rows)
    assert (forecast, board) == before


@pytest.mark.parametrize(
    "tickets,handle,qualifies",
    [(50, 65, False), (50.001, 65, True), (51, 64.999, False), (51, 65.001, True)],
)
def test_exact_thresholds(inputs, tickets, handle, qualifies):
    forecast, board, _, _ = inputs
    for book in board["games"][0]["splits"].values():
        for kind, sides in (("spread", ("home", "away")), ("total", ("over", "under"))):
            book["markets"][kind][sides[0]].update(ticket_pct=100 - tickets, handle_pct=handle)
            book["markets"][kind][sides[1]].update(ticket_pct=tickets, handle_pct=100 - handle)
    assert bool(qualifying_signals(forecast, board, now=NOW)) == qualifies


@pytest.mark.parametrize("projection,side", [(10, "home"), (-10, "away")])
def test_model_agrees_against_spread_not_winner(inputs, projection, side):
    forecast, board, _, _ = inputs
    forecast["predicted_home_margin"] = projection
    if side == "away":
        for book in board["games"][0]["splits"].values():
            values = book["markets"]["spread"]
            values["home"], values["away"] = values["away"], values["home"]
    spread = next(
        r for r in qualifying_signals(forecast, board, now=NOW) if r["market"] == "spread"
    )
    assert spread["handle_side"] == side
    forecast["predicted_home_margin"] = 3.75  # Winner favored, but no ATS edge.
    assert not any(r["market"] == "spread" for r in qualifying_signals(forecast, board, now=NOW))


def test_under_and_model_disagreement(inputs):
    forecast, board, _, _ = inputs
    forecast["total"] = 40
    assert [r["market"] for r in qualifying_signals(forecast, board, now=NOW)] == ["spread"]
    for book in board["games"][0]["splits"].values():
        values = book["markets"]["total"]
        values["over"], values["under"] = values["under"], values["over"]
    assert qualifying_signals(forecast, board, now=NOW)[1]["handle_side"] == "under"


@pytest.mark.parametrize(
    "failure",
    ["stale_split", "missing_handle", "missing_ticket", "future_split", "invalid_pair", "one_book"],
)
def test_requires_two_fresh_books_with_the_same_complete_sample(inputs, failure):
    forecast, board, _, _ = inputs
    books = board["games"][0]["splits"]
    book = books["circa"]
    if failure == "one_book":
        del books["circa"]
    elif failure in {"stale_split", "future_split"}:
        book["source_timestamp"] = (
            NOW + timedelta(seconds=-3601 if failure == "stale_split" else 1)
        ).isoformat()
    else:
        for kind, side in (("spread", "home"), ("total", "over")):
            field = "ticket_pct" if failure == "missing_ticket" else "handle_pct"
            book["markets"][kind][side][field] = 99 if failure == "invalid_pair" else None
    assert qualifying_signals(forecast, board, now=NOW) == []


def test_equal_weight_average_preserves_per_book_disagreement(inputs):
    forecast, board, _, _ = inputs
    for key, handle in (("draftkings", 90), ("circa", 40)):
        values = board["games"][0]["splits"][key]["markets"]["spread"]
        values["home"]["handle_pct"], values["away"]["handle_pct"] = handle, 100 - handle
    row = qualifying_signals(forecast, board, now=NOW)[0]
    assert row["handle_pct"] == 65
    assert row["split_books"]["circa"]["sides"]["home"]["handle_pct"] == 40


@pytest.mark.parametrize(
    "failure",
    [
        "odds_error",
        "splits_error",
        "provider_stale",
        "old_capture",
        "kickoff",
        "future_forecast",
        "missing_forecast",
        "mismatch",
    ],
)
def test_unavailable_or_postgame_inputs_never_signal(inputs, failure):
    forecast, board, _, _ = inputs
    game = board["games"][0]
    if failure in {"odds_error", "splits_error"}:
        board[failure] = "failed"
    elif failure == "provider_stale":
        game["provider_stale"] = True
    elif failure == "old_capture":
        game["captured_at"] = (NOW - timedelta(minutes=16)).isoformat()
    elif failure == "kickoff":
        game["commence_time"] = forecast["commence_time"] = STAMP
    elif failure == "future_forecast":
        forecast["forecast_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif failure == "missing_forecast":
        forecast.pop("forecast_at")
    else:
        game["home_team"] = "CLE"
    assert qualifying_signals(forecast, board, now=NOW) == []


def test_spread_and_total_freshness_are_independent(inputs):
    forecast, board, _, _ = inputs
    for book in board["games"][0]["books"].values():
        book["total_source_timestamp"] = (NOW - timedelta(minutes=16)).isoformat()
    assert [r["market"] for r in qualifying_signals(forecast, board, now=NOW)] == ["spread"]
    for book in board["games"][0]["books"].values():
        book["total_source_timestamp"] = STAMP
        book["spread_source_timestamp"] = (NOW - timedelta(minutes=16)).isoformat()
    assert [r["market"] for r in qualifying_signals(forecast, board, now=NOW)] == ["total"]


def test_worker_archives_signals_once_per_poll_and_survives_restart(tmp_path, inputs):
    forecast, _, odds, splits = inputs
    store = MarketStore(tmp_path / "market.db")
    client = Client(odds, splits)
    first = poll_market("nfl", [forecast], store=store, client=client, now=NOW)
    assert first.get("odds_error") is None
    rows = store.read_signals("nfl")["observations"]
    assert len(rows) == 2
    poll_market("nfl", [forecast], store=store, client=client, now=NOW + timedelta(seconds=30))
    assert store.read_signals("nfl")["observations"] == rows
    restarted = MarketStore(store.path)
    poll_market("nfl", [forecast], store=restarted, client=client, now=NOW + timedelta(minutes=5))
    assert len(restarted.read_signals("nfl")["observations"]) == 4
    assert restarted.read_signals("ncaaf")["observations"] == []
    with store.writer() as db:
        store.record_signals(db, rows)
    assert len(store.read_signals("nfl")["observations"]) == 4
    with sqlite3.connect(store.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM money_signals")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE money_signals SET sport='other'")


@pytest.mark.parametrize(
    "actual_margin,actual_total,status,expected",
    [
        (10, 60, "final", ["Win", "Win"]),
        (0, 30, "final", ["Loss", "Loss"]),
        (3.75, 45.5, "final", ["Push", "Push"]),
        (10, 60, "scheduled", ["Pending", "Pending"]),
        (None, None, "cancelled", ["Void", "Void"]),
    ],
)
def test_record_grades_first_signal_not_repeated_observations(
    inputs, actual_margin, actual_total, status, expected
):
    forecast, board, _, _ = inputs
    first = qualifying_signals(forecast, board, now=NOW)
    later = copy.deepcopy(first)
    for row in later:
        row.update(
            observed_at=(NOW + timedelta(minutes=5)).isoformat(),
            line=100,
            projection=0,
            signal_id="later" + row["market"],
        )
    result = pd.DataFrame(
        [
            dict(
                game_id=forecast["game_id"],
                scored_at=STAMP,
                status=status,
                actual_margin=actual_margin,
                actual_total=actual_total,
            )
        ]
    )
    rows = first_signal_records(later + first, result)
    assert rows.result.tolist() == expected
    assert len(rows) == 2
    assert rows.set_index("market").line.to_dict() == {"spread": -3.75, "total": 45.5}


def test_batch_provenance_is_attached_without_mutating_prediction(inputs):
    forecast, board, _, _ = inputs
    forecast.pop("forecast_at")
    batch = dict(
        created_at=(NOW - timedelta(hours=1)).isoformat(),
        run_id="frozen",
        model_hash="model123",
        predictions=[forecast],
    )
    enriched = published_forecasts(batch)[0]
    row = qualifying_signals(enriched, board, now=NOW)[0]
    assert row["forecast_run_id"] == "frozen" and row["model_hash"] == "model123"
    assert "forecast_created_at" not in forecast


def test_tracker_shows_current_and_preserved_history(tmp_path, inputs):
    forecast, _, odds, splits = inputs
    store = MarketStore(tmp_path / "market.db")
    poll_market("nfl", [forecast], store=store, client=Client(odds, splits), now=NOW)
    # Pin read-time evaluation so the fixture remains pregame.
    script = f"""
from datetime import datetime
from nfl_prediction import signals_ui
from nfl_prediction.current_market import MarketStore
from nfl_prediction.money_signals import qualifying_signals
signals_ui.MarketStore = lambda: MarketStore({str(store.path)!r})
signals_ui.qualifying_signals = lambda game, board: qualifying_signals(game, board, now=datetime.fromisoformat({STAMP!r}))
signals_ui.render_signal_tracker({str(tmp_path / "absent_results")!r}, sport="nfl", forecasts={json.loads(json.dumps([forecast]))!r})
"""
    app = AppTest.from_string(script).run(timeout=15)
    assert not app.exception
    assert len(app.dataframe) == 2
    assert len(app.dataframe[0].value) == len(app.dataframe[1].value) == 2
    assert app.metric[0].value == "0–0–0"
    assert app.metric[2].value == "2"


@pytest.mark.parametrize("sport", ["nfl", "ncaaf"])
def test_weekly_picks_have_requested_title_and_explanation(tmp_path, inputs, sport):
    forecast, board, _, _ = inputs
    board["games"][0]["sport"] = sport
    script = f"""
from datetime import datetime
from nfl_prediction import signals_ui
from nfl_prediction.money_signals import qualifying_signals
class Store:
    def read(self, sport): return {board!r}
signals_ui.MarketStore = Store
signals_ui.qualifying_signals = lambda game, board: qualifying_signals(game, board, now=datetime.fromisoformat({STAMP!r}))
signals_ui.render_weekly_picks({[forecast]!r}, sport={sport!r})
"""
    app = AppTest.from_string(script).run(timeout=15)
    assert not app.exception
    assert app.subheader[0].value == "Blood-in-the-water Picks of The Week"
    assert len(app.metric) == 2
    for metric in app.metric:
        assert "60.00% of tickets" in metric.proto.help
        assert "65.00% of handle" in metric.proto.help
        assert "GRIDLINE projects" in metric.proto.help
        assert "at least two fresh sportsbooks" in metric.proto.help


def test_empty_weekly_picks_do_not_fabricate_recommendations():
    script = """
from nfl_prediction import signals_ui
class Store:
    def read(self, sport): return {"games": []}
signals_ui.MarketStore = Store
signals_ui.render_weekly_picks([], sport="nfl")
"""
    app = AppTest.from_string(script).run(timeout=15)
    assert not app.exception and not app.metric
    assert any("No qualifying picks" in c.value for c in app.caption)


def test_postgres_signals_persist_and_preserve_provenance(pg_store, inputs):
    forecast, _, odds, splits = inputs
    poll_market("nfl", [forecast], store=pg_store, client=Client(odds, splits), now=NOW)
    history = MarketStore().read_signals("nfl")
    assert history["error"] is None and len(history["observations"]) == 2
    assert history["observations"][0]["split_books"]["circa"]["source_timestamp"] == STAMP
    assert MarketStore().read_signals("ncaaf")["observations"] == []


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM money_signals",
        "UPDATE money_signals SET sport='other'",
        "TRUNCATE money_signals",
    ],
)
def test_postgres_signals_are_append_only(pg_store, statement):
    import psycopg

    with pg_store.writer():
        pass
    with pytest.raises(psycopg.Error, match="append-only"), pg_store.writer() as db:
        db.execute(statement)


def test_total_quote_timestamp_is_not_borrowed_from_spread(inputs):
    forecast, _, odds, _ = inputs
    for events in odds["data"].values():
        markets = events[0]["bookmakers"][0]["markets"]
        markets[0]["last_update"] = STAMP
        markets[1]["last_update"] = (NOW - timedelta(minutes=30)).isoformat()
    game = parse_odds(odds, "nfl", [forecast], STAMP)["games"][0]
    assert all(b["spread_source_timestamp"] == STAMP for b in game["books"].values())
    assert all(
        b["total_source_timestamp"] == (NOW - timedelta(minutes=30)).isoformat()
        for b in game["books"].values()
    )
