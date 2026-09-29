from __future__ import annotations

import copy
import sqlite3
from datetime import timedelta

import pandas as pd
import psycopg
import pytest
import test_market_deployment
import test_owls
from test_owls import EVENT, KICKOFF, NOW, STAMP

from nfl_prediction.current_market import MarketStore
from nfl_prediction.owls import OwlsError
from nfl_prediction.player_props import comparisons, parse_props, player_name_key, projection_rows
from nfl_prediction.prop_cache import parse_depth, poll_depth, poll_props

slate = test_owls.slate
pg_store = test_market_deployment.pg_store


@pytest.fixture
def payload():
    return {
        "success": True,
        "data": [
            {
                "gameId": EVENT,
                "homeTeam": "Buffalo Bills",
                "awayTeam": "New York Jets",
                "commenceTime": KICKOFF,
                "isLive": False,
                "books": [
                    {
                        "key": "draftkings",
                        "lastUpdate": STAMP,
                        "props": [
                            {
                                "playerName": "Josh Allen",
                                "team": "Buffalo Bills",
                                "category": "passing_yards",
                                "line": 250.5,
                                "overPrice": -110,
                                "underPrice": -115,
                                "alternateLines": [{"line": 300.5, "odds": 200}],
                            }
                        ],
                    }
                ],
            }
        ],
    }


@pytest.fixture
def projections(slate):
    return [
        {
            "player_id": "gsis-1",
            "player_name": "J.Allen",
            "team": "BUF",
            "opponent": "NYJ",
            "game_id": slate[0]["game_id"],
            "commence_time": KICKOFF,
            "category": "passing_yards",
            "projection": 270.0,
        }
    ]


@pytest.fixture
def depth():
    return {
        "players": [
            {
                "team": "BUF",
                "player_id": "gsis-1",
                "player_name": "Josh Allen",
                "position": "QB",
                "starter": True,
                "source_timestamp": STAMP,
            }
        ]
    }


class Client:
    quota_retry_after = 0

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, sport, endpoint):
        self.calls.append((sport, endpoint))
        if isinstance(self.payload, Exception):
            raise self.payload
        return copy.deepcopy(self.payload)


def test_main_lines_parse_without_alternates_and_normalize_team(payload, slate):
    before = copy.deepcopy(payload)
    board = parse_props(payload, slate, STAMP)
    (row,) = board["rows"]
    assert (row["line"], row["team"], row["event_id"]) == (250.5, "BUF", EVENT)
    assert row["source_timestamp"] == STAMP
    assert row["game_id"] == slate[0]["game_id"]
    assert payload == before
    assert player_name_key("Odell Beckham Jr.") == player_name_key("Odell Beckham")
    assert player_name_key("D’Andre Swift") == player_name_key("D'Andre Swift")
    assert player_name_key("J.Allen") != player_name_key("Josh Allen")


@pytest.mark.parametrize(
    "change",
    [
        {"marketStructure": "milestone"},
        {"marketStructure": "unknown"},
        {"line": 250},
        {"isMain": False},
        {"isAlternate": True},
        {"overPrice": None},
        {"underPrice": None},
        {"underPrice": 0},
        {"line": float("nan")},
        {"line": -1.5},
        {"playerName": ""},
        {"team": "Miami Dolphins"},
    ],
)
def test_unsafe_or_ambiguous_quotes_are_excluded(payload, slate, change):
    payload["data"][0]["books"][0]["props"][0].update(change)
    board = parse_props(payload, slate, STAMP)
    assert not board["rows"] and board["diagnostics"]


def test_explicit_over_under_can_have_integer_line(payload, slate):
    payload["data"][0]["books"][0]["props"][0].update(line=250, marketStructure="over_under")
    assert parse_props(payload, slate, STAMP)["rows"][0]["line"] == 250


def test_conflicting_main_lines_reject_both(payload, slate):
    props = payload["data"][0]["books"][0]["props"]
    props.append({**props[0], "line": 255.5})
    board = parse_props(payload, slate, STAMP)
    assert not board["rows"]
    assert board["diagnostics"]["conflicting_main_lines"] == 1


def test_provider_stale_flag_is_respected(payload, slate, projections, depth):
    payload["meta"] = {"books": {"draftkings": {"status": "stale"}}}
    board = parse_props(payload, slate, STAMP)
    assert board["rows"][0]["provider_stale"]
    assert not comparisons(projections, board, depth, "passing_yards", now=NOW)


def test_malformed_categories_do_not_crash_board(payload, slate):
    payload["data"][0]["books"][0]["props"][0]["category"] = {}
    assert not parse_props(payload, slate, STAMP)["rows"]


def test_invalid_slate_reports_kickoff_problem_without_provider_call(tmp_path, payload, slate):
    slate[0]["commence_time"] = "invalid"
    client = Client(payload)
    result = poll_props(slate, store=MarketStore(tmp_path / "props.db"), client=client, now=NOW)
    assert "kickoff" in result["error"] and not client.calls


@pytest.mark.parametrize(
    "change", [{"homeTeam": "Unknown"}, {"commenceTime": STAMP}, {"isLive": True}, {"gameId": ""}]
)
def test_event_matching_requires_teams_kickoff_and_pregame(payload, slate, change):
    payload["data"][0].update(change)
    assert not parse_props(payload, slate, STAMP)["rows"]


@pytest.mark.parametrize(
    "payload", [{}, {"data": {}}, {"data": [None]}, {"data": [{"books": None}]}]
)
def test_malformed_boards_raise_redacted_error(payload, slate):
    with pytest.raises(OwlsError, match="malformed"):
        parse_props(payload, slate, STAMP)


def test_comparison_median_sorting_and_market_updates_do_not_change_projections(
    payload, slate, projections, depth
):
    board = parse_props(payload, slate, STAMP)
    quote = board["rows"][0]
    board["rows"].append({**quote, "sportsbook": "fanduel", "line": 260.5})
    projections.append({**projections[0], "player_id": "gsis-2", "projection": 210.0})
    depth["players"].append(
        {**depth["players"][0], "player_id": "gsis-2", "player_name": "Other Starter"}
    )
    board["rows"].append({**quote, "player_key": "otherstarter", "player_name": "Other Starter"})
    original = copy.deepcopy(projections)
    rows = comparisons(projections, board, depth, "passing_yards", now=NOW)
    assert [r["difference"] for r in rows] == [-40.5, 14.5]
    assert [r["lean"] for r in rows] == ["Under", "Over"]
    assert rows[1]["books"] == 2 and rows[1]["line"] == 255.5
    single = comparisons(projections, board, depth, "passing_yards", book="fanduel", now=NOW)
    assert single[0]["line"] == 260.5 and single[0]["difference"] == 9.5
    quote["line"] = 270.0
    assert (
        comparisons(projections, board, depth, "passing_yards", book="draftkings", now=NOW)[1][
            "lean"
        ]
        == "Even"
    )
    assert projections == original
    assert not comparisons(projections, board, depth, "receptions", now=NOW)


@pytest.mark.parametrize(
    "change",
    ["backup", "old_depth", "missing_id", "wrong_team", "initial_only", "ambiguous", "started"],
)
def test_only_identified_upcoming_expected_starters(payload, slate, projections, depth, change):
    board = parse_props(payload, slate, STAMP)
    now = NOW
    if change == "backup":
        depth["players"][0]["starter"] = False
    elif change == "old_depth":
        depth["players"][0]["source_timestamp"] = (NOW - timedelta(hours=49)).isoformat()
    elif change == "missing_id":
        depth["players"][0]["player_id"] = "different-id"
    elif change == "wrong_team":
        depth["players"][0]["team"] = "NYJ"
    elif change == "initial_only":
        depth["players"][0]["player_name"] = "J.Allen"
    elif change == "ambiguous":
        projections.append({**projections[0], "player_id": "different-id"})
        depth["players"].append({**depth["players"][0], "player_id": "different-id"})
    else:
        now = NOW + timedelta(days=4)
    assert not comparisons(projections, board, depth, "passing_yards", now=now)


@pytest.mark.parametrize(
    "change", ["error", "old_source", "old_capture", "missing_time", "future_time", "unavailable"]
)
def test_old_quotes_require_opt_in_and_remain_labeled(payload, slate, projections, depth, change):
    board = parse_props(payload, slate, STAMP)
    quote = board["rows"][0]
    if change == "error":
        board["error"] = "Owls props: rate limit"
    elif change == "old_source":
        quote["source_timestamp"] = (NOW - timedelta(minutes=61)).isoformat()
    elif change == "old_capture":
        quote["captured_at"] = (NOW - timedelta(minutes=61)).isoformat()
    elif change == "missing_time":
        quote["source_timestamp"] = None
    elif change == "future_time":
        quote["source_timestamp"] = (NOW + timedelta(hours=2)).isoformat()
    else:
        quote["unavailable"] = True
    assert not comparisons(projections, board, depth, "passing_yards", now=NOW)
    (row,) = comparisons(projections, board, depth, "passing_yards", include_stale=True, now=NOW)
    assert row["stale"] and row["projection"] == 270


def test_depth_chart_uses_latest_team_and_each_receiver_slot():
    base = {
        "dt": STAMP,
        "team": "BUF",
        "player_name": "Receiver",
        "gsis_id": "one",
        "pos_abb": "WR",
        "pos_grp": "WR",
        "pos_slot": 1,
        "pos_rank": 1,
    }
    frame = pd.DataFrame(
        [
            base,
            {**base, "gsis_id": "two", "pos_slot": 2, "pos_rank": 2},
            {**base, "gsis_id": "three", "pos_slot": 8, "pos_rank": 3},
            {**base, "gsis_id": "backup", "pos_rank": 4},
            {**base, "gsis_id": "old", "dt": (NOW - timedelta(days=1)).isoformat()},
            {**base, "gsis_id": "future", "dt": (NOW + timedelta(days=1)).isoformat()},
        ]
    )
    players = {p["player_id"]: p for p in parse_depth(frame, NOW)["players"]}
    assert set(players) == {"one", "two", "three", "backup"}
    assert all(players[p]["starter"] for p in ("one", "two", "three"))
    assert not players["backup"]["starter"]
    frame.loc[3, "pos_rank"] = 1
    assert not next(p for p in parse_depth(frame, NOW)["players"] if p["player_id"] == "one")[
        "starter"
    ]
    with pytest.raises(OwlsError, match="schema"):
        parse_depth(pd.DataFrame(), NOW)


def test_unknown_starter_id_does_not_promote_backup():
    base = {
        "dt": STAMP,
        "team": "BUF",
        "player_name": "Unknown Starter",
        "gsis_id": None,
        "pos_abb": "QB",
        "pos_grp": "QB",
        "pos_slot": 1,
        "pos_rank": 1,
    }
    players = parse_depth(
        pd.DataFrame([base, {**base, "player_name": "Backup", "gsis_id": "backup", "pos_rank": 2}]),
        NOW,
    )["players"]
    assert len(players) == 1 and not players[0]["starter"]


def test_projection_generation_uses_existing_snapshot_and_no_market_inputs(slate):
    class Model:
        def distribution(self, frame):
            assert list(frame.columns) == list(player)
            return [{"mean": 270}]

    player = {
        "player_name": "J.Allen",
        "team": "BUF",
        "opponent": "NYJ",
        "prediction_season": 2026,
        "roster_season": 2026,
    }
    state = {"qb": {"gsis-1": player}, "schedule": slate}
    before = copy.deepcopy(state)
    rows = projection_rows(state, {"passing_yards": Model()}, {"prediction_season": 2026})
    assert rows[0]["projection"] == 270 and state == before
    player["roster_season"] = 2025
    assert not projection_rows(state, {"passing_yards": Model()}, {"prediction_season": 2026})


def test_persisted_cadence_history_and_return_to_original_line(tmp_path, payload, slate):
    path = tmp_path / "props.db"
    client = Client(payload)
    for minute, line in [(0, 250.5), (1, 250.5), (15, 250.5), (30, 260.5), (45, 250.5)]:
        client.payload["data"][0]["books"][0]["props"][0]["line"] = line
        result = poll_props(
            slate, store=MarketStore(path), client=client, now=NOW + timedelta(minutes=minute)
        )
        assert not result["error"]
    assert len(client.calls) == 4
    with MarketStore(path).writer() as db:
        assert db.execute("SELECT count(*) FROM prop_observations").fetchone()[0] == 3
    for sql in ("DELETE FROM prop_observations", "UPDATE prop_observations SET category='bad'"):
        with (
            pytest.raises(sqlite3.IntegrityError, match="append-only"),
            MarketStore(path).writer() as db,
        ):
            db.execute(sql)


def test_rate_failure_preserves_cache_and_sets_shared_backoff(tmp_path, payload, slate):
    store = MarketStore(tmp_path / "props.db")
    client = Client(payload)
    good = poll_props(slate, store=store, client=client, now=NOW)
    client.payload = OwlsError("Owls props: rate limit", retry_after=1800)
    failed = poll_props(slate, store=store, client=client, now=NOW + timedelta(minutes=15))
    assert failed["rows"] == good["rows"] and "rate limit" in failed["error"]
    poll_props(slate, store=store, client=client, now=NOW + timedelta(minutes=16))
    assert len(client.calls) == 2
    with store.writer() as db:
        assert (
            db.execute("SELECT next_at FROM poll_state WHERE key='account'").fetchone()[0]
            == (NOW + timedelta(minutes=45)).isoformat()
        )


def test_other_endpoint_backoff_is_visible_to_readers(tmp_path, payload, slate):
    store = MarketStore(tmp_path / "props.db")
    client = Client(payload)
    poll_props(slate, store=store, client=client, now=NOW)
    with store.writer() as db:
        db.execute(
            "INSERT INTO poll_state VALUES ('account',?)", ((NOW + timedelta(hours=1)).isoformat(),)
        )
    poll_props(slate, store=store, client=client, now=NOW + timedelta(minutes=1))
    assert "backoff" in store.read("nfl_props")["error"]
    assert len(client.calls) == 1


def test_missing_quotes_retained_as_unavailable_no_upcoming_avoids_calls(tmp_path, payload, slate):
    store = MarketStore(tmp_path / "props.db")
    client = Client(payload)
    poll_props(slate, store=store, client=client, now=NOW)
    client.payload = {"success": True, "data": []}
    result = poll_props(slate, store=store, client=client, now=NOW + timedelta(minutes=15))
    assert result["rows"][0]["unavailable"]
    result = poll_props(slate, store=store, client=client, now=NOW + timedelta(days=6))
    assert not result["rows"] and len(client.calls) == 2
    assert result["diagnostics"]["no_upcoming_forecasts"] == 1


def test_depth_refresh_cadence_and_failure_keeps_source_age(tmp_path, monkeypatch, depth):
    calls = []

    def fetch(season, now):
        calls.append(season)
        if len(calls) > 1:
            raise OwlsError("Depth chart refresh unavailable")
        return copy.deepcopy(depth)

    monkeypatch.setattr("nfl_prediction.prop_cache.fetch_depth", fetch)
    store = MarketStore(tmp_path / "props.db")
    poll_depth(2026, store=store, now=NOW)
    poll_depth(2026, store=store, now=NOW + timedelta(hours=1))
    result = poll_depth(2026, store=store, now=NOW + timedelta(hours=6))
    assert len(calls) == 2 and result["players"] == depth["players"]
    assert result["error"]


def test_postgres_props_shared_cache_and_immutable_history(pg_store, payload, slate):
    client = Client(payload)
    board = poll_props(slate, store=pg_store, client=client, now=NOW)
    assert not board["error"]
    assert MarketStore().read("nfl_props") == board
    poll_props(slate, store=MarketStore(), client=client, now=NOW + timedelta(minutes=1))
    assert len(client.calls) == 1
    for sql in (
        "DELETE FROM prop_observations",
        "UPDATE prop_observations SET category='bad'",
        "TRUNCATE prop_observations",
    ):
        with pytest.raises(psycopg.Error, match="append-only"), pg_store.writer() as db:
            db.execute(sql)
