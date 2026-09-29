from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event

import psycopg
import pytest
import test_owls
from psycopg.conninfo import conninfo_to_dict
from test_owls import NOW, Client

import current_market_update
import serve
from nfl_prediction.current_market import MarketStore, current_context, poll_market
from nfl_prediction.owls import OwlsError

odds = test_owls.odds
slate = test_owls.slate
splits = test_owls.splits


@pytest.fixture
def pg_store(monkeypatch):
    url = os.environ.get("GRIDLINE_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "Set GRIDLINE_TEST_DATABASE_URL to the disposable gridline_market_test database"
        )
    # Never clear a production database when somebody supplies the wrong URL.
    assert conninfo_to_dict(url).get("dbname") == "gridline_market_test"
    monkeypatch.setenv("GRIDLINE_MARKET_DATABASE_URL", url)
    with psycopg.connect(url, autocommit=True) as db:
        db.execute("DROP SCHEMA IF EXISTS gridline_market CASCADE")
    yield MarketStore()
    with psycopg.connect(url, autocommit=True) as db:
        db.execute("DROP SCHEMA IF EXISTS gridline_market CASCADE")


def test_heroku_requires_persistent_database(monkeypatch, tmp_path, odds, splits, slate):
    monkeypatch.setenv("DYNO", "web.1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("GRIDLINE_MARKET_DATABASE_URL", raising=False)
    store = MarketStore(tmp_path / "ephemeral.db")
    client = Client(odds, splits)
    board = poll_market("nfl", slate, store=store, client=client, now=NOW)
    assert "Persistent market database" in board["odds_error"]
    assert store.read("nfl")["odds_error"] == board["odds_error"]
    assert not client.calls and not store.path.exists()


def test_database_selection_and_no_secret_in_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret-default")
    assert MarketStore().database_url == "postgresql://secret-default"
    monkeypatch.setenv("GRIDLINE_MARKET_DATABASE_URL", "postgresql://secret-market")
    store = MarketStore()
    assert store.database_url == "postgresql://secret-market"
    assert MarketStore(tmp_path / "local.db").database_url is None

    def fail(*args):
        raise psycopg.OperationalError("secret-market")

    monkeypatch.setattr("nfl_prediction.market_postgres.read", fail)
    assert store.read("nfl") == {"games": [], "odds_error": "Market cache unavailable"}


def test_deployment_status_never_fetches_or_prints_credentials(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OWLS_INSIGHT_API_KEY", "status-secret")
    monkeypatch.delenv("DYNO", raising=False)

    def fail(*args, **kwargs):
        pytest.fail("Status must never call the provider")

    monkeypatch.setattr(current_market_update, "poll_market", fail)
    monkeypatch.setattr(current_market_update, "poll_props", fail)
    monkeypatch.setattr(current_market_update, "poll_depth", fail)
    assert current_market_update.main(["--status", "--db", str(tmp_path / "empty.db")]) == 0
    output = capsys.readouterr().out
    assert "status-secret" not in output
    status = json.loads(output)
    assert status["owls_key_configured"] is True
    assert status["sports"]["nfl"]["cached_games"] == 0


def test_postgres_separate_workers_readers_and_restart(pg_store, odds, splits, slate):
    original = copy.deepcopy(slate)
    client = Client(odds, splits)
    board = poll_market("nfl", slate, store=pg_store, client=client, now=NOW)
    assert board.get("odds_error") is None
    reader = MarketStore()
    assert reader.read("nfl") == board
    assert current_context(slate[0], reader.read("nfl"), now=NOW)["home_spread"] == -3.75
    poll_market("nfl", slate, store=MarketStore(), client=client, now=NOW + timedelta(seconds=30))
    assert len(client.calls) == 2
    poll_market("nfl", slate, store=MarketStore(), client=client, now=NOW + timedelta(minutes=5))
    with pg_store.writer() as db:
        assert db.execute("SELECT count(*) FROM market_observations").fetchone()[0] == 2
    outcomes = client.odds["data"]["draftkings"][0]["bookmakers"][0]["markets"][0]["outcomes"]
    outcomes[0]["point"], outcomes[1]["point"] = -5, 5
    poll_market("nfl", slate, store=MarketStore(), client=client, now=NOW + timedelta(minutes=10))
    with pg_store.writer() as db:
        assert db.execute("SELECT count(*) FROM market_observations").fetchone()[0] == 3
    assert MarketStore().read("nfl")["games"][0]["spread"]["home_spread"] == -4.5
    assert slate == original


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM market_observations",
        "UPDATE market_observations SET sport='changed'",
        "TRUNCATE market_observations",
    ],
)
def test_postgres_history_is_append_only(pg_store, odds, splits, slate, statement):
    poll_market("nfl", slate, store=pg_store, client=Client(odds, splits), now=NOW)
    with pytest.raises(psycopg.Error, match="append-only"), pg_store.writer() as db:
        db.execute(statement)


def test_postgres_shared_account_backoff(pg_store, odds, splits, slate):
    client = Client(odds, splits)
    poll_market("nfl", slate, store=pg_store, client=client, now=NOW)
    client.odds = OwlsError("rate limit", retry_after=1800)
    failed = poll_market(
        "nfl", slate, store=pg_store, client=client, now=NOW + timedelta(minutes=5)
    )
    assert current_context(slate[0], failed, now=NOW + timedelta(minutes=5))["status"] == "stale"
    calls = len(client.calls)
    poll_market("ncaaf", slate, store=MarketStore(), client=client, now=NOW + timedelta(minutes=6))
    assert len(client.calls) == calls
    assert MarketStore().read("nfl")["games"] == failed["games"]


def test_postgres_concurrent_pollers_only_fetch_once(pg_store, odds, splits, slate):
    client = Client(odds, splits)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: poll_market("nfl", slate, store=MarketStore(), client=client, now=NOW),
                range(2),
            )
        )
    assert all(result.get("odds_error") is None for result in results)
    assert len(client.calls) == 2


class Process:
    def __init__(self, code=None):
        self.code = code
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 0

    def wait(self, timeout):
        return self.code


def test_web_starts_worker_without_browser_session_and_stops_both(monkeypatch):
    processes, commands = [], []
    stop = Event()

    def launch(command, **kwargs):
        commands.append(command)
        process = Process()
        processes.append(process)
        if len(processes) == 2:
            stop.set()
        return process

    monkeypatch.setenv("PORT", "12345")
    monkeypatch.delenv("GRIDLINE_MARKET_AUTOSTART", raising=False)
    monkeypatch.setattr(serve.subprocess, "Popen", launch)
    assert serve.run(stop) == 0
    assert "--server.port=12345" in commands[0]
    assert commands[1][-2:] == ["current_market_update.py", "--watch"]
    assert all(process.terminated for process in processes)


def test_worker_failure_does_not_stop_forecasts(monkeypatch):
    stop = Event()
    processes = []
    ticks = iter([0, 0, 0, 61])

    def launch(command, **kwargs):
        process = Process(1 if len(processes) == 1 else None)
        processes.append(process)
        if len(processes) == 3:
            stop.set()
        return process

    monkeypatch.delenv("GRIDLINE_MARKET_AUTOSTART", raising=False)
    monkeypatch.setattr(serve.subprocess, "Popen", launch)
    monkeypatch.setattr(serve.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(stop, "wait", lambda _: None)
    assert serve.run(stop) == 0
    assert len(processes) == 3
    assert processes[0].terminated  # Only when the service is shut down.


def test_web_exit_stops_worker(monkeypatch):
    stop = Event()
    web = Process()
    worker = Process()

    def launch(command, **kwargs):
        if "--watch" in command:
            web.code = 7
            return worker
        return web

    monkeypatch.delenv("GRIDLINE_MARKET_AUTOSTART", raising=False)
    monkeypatch.setattr(serve.subprocess, "Popen", launch)
    monkeypatch.setattr(stop, "wait", lambda _: None)
    assert serve.run(stop) == 7
    assert worker.terminated
