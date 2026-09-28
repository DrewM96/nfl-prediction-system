from __future__ import annotations

import copy
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError

import pytest

from nfl_prediction.current_market import (
    MarketStore,
    current_context,
    freeze_market_context,
    poll_market,
)
from nfl_prediction.market_parity import compare_boards
from nfl_prediction.market_ui import market_context_html, public_action_html
from nfl_prediction.odds import CreditUsage, OddsFetchResult, build_consensus
from nfl_prediction.owls import OwlsClient, OwlsError, match_event, parse_odds, parse_splits

NOW = datetime(2026, 9, 10, 16, tzinfo=UTC)
STAMP = NOW.isoformat()
KICKOFF = "2026-09-13T17:00:00+00:00"
EVENT = "nfl:New York Jets@Buffalo Bills-20260913"


@pytest.fixture
def slate():
    return [
        {
            "game_id": "2026_02_NYJ_BUF",
            "home_team": "BUF",
            "away_team": "NYJ",
            "commence_time": KICKOFF,
            "predicted_home_margin": 2.0,
            "forecast_at": "2026-09-10T15:00:00+00:00",
            "market_line": -3.0,
            "market_consensus": {
                "provider": "original",
                "snapshot_at": "2026-09-10T15:00:00+00:00",
                "spread": {"home_spread": -3.0},
            },
        }
    ]


@pytest.fixture
def odds():
    def event(book, spread):
        return {
            "id": f"local-{book}",
            "eventId": EVENT,
            "home_team": "Buffalo Bills",
            "away_team": "New York Jets",
            "commence_time": KICKOFF,
            "bookmakers": [
                {
                    "key": book,
                    "last_update": STAMP,
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {
                                    "name": "Buffalo Bills",
                                    "point": spread,
                                    "price": -110,
                                    "alternateLines": [{"point": -17.5, "price": 400}],
                                },
                                {"name": "New York Jets", "point": -spread, "price": -110},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 45.5, "price": -105},
                                {"name": "Under", "point": 45.5, "price": -115},
                            ],
                        },
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Buffalo Bills", "price": -150},
                                {"name": "New York Jets", "price": 130},
                            ],
                        },
                    ],
                }
            ],
        }

    return {
        "success": True,
        "data": {"draftkings": [event("draftkings", -3.5)], "circa": [event("circa", -4.0)]},
        "meta": {"timestamp": STAMP, "freshness": {"stale": False}},
    }


@pytest.fixture
def splits():
    return {
        "sport": "nfl",
        "data": [
            {
                "event_id": EVENT,
                "home_team": "Buffalo Bills",
                "away_team": "New York Jets",
                "splits": [
                    {
                        "book": "dk",
                        "as_of": STAMP,
                        "spread": {
                            "home_bets_pct": 60,
                            "away_bets_pct": 40,
                            "home_handle_pct": 45,
                            "away_handle_pct": 55,
                        },
                        "moneyline": {"home_bets_pct": 70, "away_bets_pct": 30},
                        "total": {
                            "over_bets_pct": 62,
                            "under_bets_pct": 38,
                            "over_handle_pct": 71,
                            "under_handle_pct": 29,
                        },
                    },
                    {
                        "book": "circa",
                        "as_of": STAMP,
                        "spread": {
                            "home_bets_pct": 40,
                            "away_bets_pct": 60,
                            "home_handle_pct": 65,
                            "away_handle_pct": 35,
                        },
                    },
                ],
            }
        ],
    }


class Client:
    def __init__(self, odds, splits):
        self.odds, self.splits = odds, splits
        self.calls = []

    def get(self, sport, endpoint):
        self.calls.append((sport, endpoint))
        value = getattr(self, endpoint)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def test_normalized_board_preserves_consensus_and_prices(odds, slate):
    board = parse_odds(odds, "nfl", slate, STAMP)
    game = board["games"][0]
    assert game["spread"]["home_spread"] == -3.75
    assert game["spread"]["market_home_margin"] == 3.75
    assert game["spread"]["line_iqr"] == 0.25
    assert game["spread"]["book_count"] == 2
    assert game["books"]["draftkings"]["moneyline"] == -150
    assert game["total"]["total"] == 45.5
    event = copy.deepcopy(odds["data"]["draftkings"][0])
    event["bookmakers"] += odds["data"]["circa"][0]["bookmakers"]
    legacy = build_consensus(
        OddsFetchResult([event], STAMP, CreditUsage(), "current"),
        regions="us",
        markets=("spreads", "totals"),
    )
    assert game["spread"] == legacy["games"][0]["spread"]


def test_duplicate_book_and_alternates_do_not_weight_consensus(odds, slate):
    odds["data"]["draftkings"].append(copy.deepcopy(odds["data"]["draftkings"][0]))
    board = parse_odds(odds, "nfl", slate, STAMP)
    assert board["games"][0]["spread"]["book_count"] == 2
    assert board["diagnostics"][0]["reason"] == "duplicate_book"


@pytest.mark.parametrize(
    "home,away", [("Texas A&M Aggies", "Miami (OH) RedHawks"), ("Texas A&M", "Miami Ohio")]
)
def test_cfb_explicit_names_and_event_identity(home, away):
    slate = [
        {"game_id": 123, "home_team": "Texas A&M", "away_team": "Miami (OH)", "start_date": KICKOFF}
    ]
    event = {"home_team": home, "away_team": away, "commence_time": KICKOFF}
    assert match_event(event, "ncaaf", slate)[0] == slate[0]
    event["commence_time"] = "2026-09-20T17:00:00Z"
    assert match_event(event, "ncaaf", slate)[1] == "slate_or_kickoff_mismatch"
    event["home_team"] = "Texas Unknown"
    assert match_event(event, "ncaaf", slate)[1] == "unknown_team"


def test_ambiguous_game_and_team_mismatch_are_reported(odds, slate):
    duplicate = [{**slate[0], "game_id": "other"}]
    board = parse_odds(odds, "nfl", slate + duplicate, STAMP)
    assert not board["games"]
    assert {d["reason"] for d in board["diagnostics"]} == {"ambiguous_game"}


def test_splits_are_book_specific_missing_is_not_zero(splits):
    parsed = parse_splits(splits)[EVENT]
    assert set(parsed) == {"draftkings", "circa"}
    spread = parsed["draftkings"]["markets"]["spread"]
    assert spread["home"]["handle_minus_ticket"] == -15
    assert spread["majority_ticket_side"] == "home"
    assert spread["majority_money_side"] == "away"
    assert spread["ticket_money_disagreement"] is True
    assert parsed["draftkings"]["markets"]["moneyline"]["home"]["handle_pct"] is None
    assert parsed["circa"]["markets"]["total"]["over"]["ticket_pct"] is None


def test_invalid_split_percentages_and_ties(splits):
    raw = splits["data"][0]["splits"][0]["spread"]
    raw.update(home_bets_pct=50, away_bets_pct=50, home_handle_pct=101, away_handle_pct=-1)
    market = parse_splits(splits)[EVENT]["draftkings"]["markets"]["spread"]
    assert market["majority_ticket_side"] is None
    assert market["home"]["handle_pct"] is None
    assert market["ticket_money_disagreement"] is None


def test_cache_history_edge_and_frozen_separation(tmp_path, odds, splits, slate):
    store = MarketStore(tmp_path / "market.db")
    client = Client(odds, splits)
    original = copy.deepcopy(slate)
    board = poll_market("nfl", slate, store=store, client=client, now=NOW)
    context = current_context(slate[0], board, now=NOW)
    assert context["status"] == "fresh"
    assert context["movement"] == -0.75
    assert context["current_home_edge"] == -1.75
    assert context["sportsbook_agreement"]["spread"]["majority_ticket_side"] == "disagree"
    assert slate == original
    poll_market("nfl", slate, store=store, client=client, now=NOW + timedelta(seconds=30))
    assert len(client.calls) == 2
    poll_market("nfl", slate, store=store, client=client, now=NOW + timedelta(minutes=5))
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM market_observations").fetchone()[0] == 2
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM market_observations")
    client.odds["data"]["draftkings"][0]["bookmakers"][0]["markets"][0]["outcomes"][0]["point"] = -5
    client.odds["data"]["draftkings"][0]["bookmakers"][0]["markets"][0]["outcomes"][1]["point"] = 5
    changed = poll_market("nfl", slate, store=store, client=client, now=NOW + timedelta(minutes=10))
    assert (
        current_context(slate[0], changed, now=NOW + timedelta(minutes=10))["current_home_edge"]
        == -2.5
    )
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM market_observations").fetchone()[0] == 3
    assert slate == original


def test_freeze_captures_once_without_changing_projection(
    tmp_path, odds, splits, slate, monkeypatch
):
    monkeypatch.setenv("GRIDLINE_MARKET_PROVIDER", "owls")
    store = MarketStore(tmp_path / "market.db")
    poll_market("nfl", slate, store=store, client=Client(odds, splits), now=NOW)
    frozen = freeze_market_context(slate, "nfl", as_of=NOW, store=store)[0]
    assert frozen["predicted_home_margin"] == slate[0]["predicted_home_margin"]
    assert frozen["market_line"] == -3.75
    assert frozen["market_consensus"]["event_id"] == EVENT
    assert frozen["forecast_at"] == STAMP
    assert slate[0]["market_line"] == -3
    unavailable = freeze_market_context(slate, "nfl", as_of=NOW + timedelta(hours=1), store=store)[
        0
    ]
    assert unavailable["market_consensus"] is None
    assert unavailable["predicted_home_margin"] == 2


def test_failed_odds_retains_cache_and_backoff_survives_restart(tmp_path, odds, splits, slate):
    store = MarketStore(tmp_path / "market.db")
    client = Client(odds, splits)
    poll_market("nfl", slate, store=store, client=client, now=NOW)
    client.odds = OwlsError("rate limit", retry_after=1800)
    failed = poll_market("nfl", slate, store=store, client=client, now=NOW + timedelta(minutes=5))
    context = current_context(slate[0], failed, now=NOW + timedelta(minutes=5))
    assert context["status"] == "stale"
    assert context["home_spread"] == -3.75
    assert context["captured_at"] == STAMP
    calls = len(client.calls)
    poll_market(
        "nfl", slate, store=MarketStore(store.path), client=client, now=NOW + timedelta(minutes=11)
    )
    poll_market("ncaaf", slate, store=store, client=client, now=NOW + timedelta(minutes=11))
    assert len(client.calls) == calls


def test_missing_game_and_missing_book_splits_are_explicit(tmp_path, odds, splits, slate):
    store = MarketStore(tmp_path / "market.db")
    client = Client(odds, splits)
    poll_market("nfl", slate, store=store, client=client, now=NOW)
    client.odds = {"data": {}}
    client.splits = {"data": []}
    board = poll_market("nfl", slate, store=store, client=client, now=NOW + timedelta(minutes=5))
    context = current_context(slate[0], board, now=NOW + timedelta(minutes=5))
    assert context["status"] == "stale"
    assert "Game missing" in context["reason"]
    assert context["splits"]["draftkings"]["stale"]


def test_failed_splits_do_not_hide_fresh_odds(tmp_path, odds, splits, slate):
    store = MarketStore(tmp_path / "market.db")
    client = Client(odds, splits)
    poll_market("nfl", slate, store=store, client=client, now=NOW)
    client.splits = OwlsError("splits unavailable")
    board = poll_market("nfl", slate, store=store, client=client, now=NOW + timedelta(minutes=5))
    context = current_context(slate[0], board, now=NOW + timedelta(minutes=5))
    assert context["status"] == "fresh"
    assert context["splits"]["circa"]["stale"]


@pytest.mark.parametrize(
    "timestamp_value", [None, "invalid", "2026-09-10T12:00:00Z", "2026-09-11T16:00:00Z"]
)
def test_stale_source_never_relabelled_by_fresh_capture(odds, slate, timestamp_value):
    odds["data"]["draftkings"][0]["bookmakers"][0]["last_update"] = timestamp_value
    board = parse_odds(odds, "nfl", slate, STAMP)
    assert current_context(slate[0], board, now=NOW)["status"] == "stale"


def test_unavailable_cache_and_mismatched_kickoff_do_not_crash(tmp_path, odds, slate):
    assert (
        current_context(slate[0], MarketStore(tmp_path / "absent").read("nfl"), now=NOW)["status"]
        == "unavailable"
    )
    board = parse_odds(odds, "nfl", slate, STAMP)
    slate[0]["commence_time"] = "2026-09-20T17:00:00Z"
    assert current_context(slate[0], board, now=NOW)["status"] == "unavailable"


@pytest.mark.parametrize(
    "payload", [{}, {"data": []}, {"data": {"book": None}}, {"success": False, "data": {}}]
)
def test_malformed_board_is_explicit(payload, slate):
    with pytest.raises(OwlsError):
        parse_odds(payload, "nfl", slate, STAMP)


@pytest.mark.parametrize("code", [401, 403, 429, 500])
def test_redacted_http_errors_and_retry_after(monkeypatch, code):
    monkeypatch.setenv("OWLS_INSIGHT_API_KEY", "secret-never-log")

    def fail(request, **kwargs):
        assert request.get_header("Authorization") == "Bearer secret-never-log"
        assert "secret" not in request.full_url
        raise HTTPError(request.full_url, code, "secret-never-log", {"Retry-After": "1900"}, None)

    monkeypatch.setattr("nfl_prediction.owls.urlopen", fail)
    with pytest.raises(OwlsError) as caught:
        OwlsClient().get("nfl", "odds")
    assert "secret" not in str(caught.value)
    assert caught.value.retry_after >= 1900


def test_key_is_environment_only(monkeypatch):
    monkeypatch.delenv("OWLS_INSIGHT_API_KEY", raising=False)
    with pytest.raises(OwlsError, match="not configured"):
        OwlsClient().get("ncaaf", "splits")


def test_network_and_malformed_json_failures(monkeypatch):
    monkeypatch.setenv("OWLS_INSIGHT_API_KEY", "test")

    def fail(*args, **kwargs):
        raise URLError("secret")

    monkeypatch.setattr("nfl_prediction.owls.urlopen", fail)
    with pytest.raises(OwlsError, match="network failure"):
        OwlsClient().get("nfl", "odds")


def test_provider_parity_reports_coverage_spreads_and_normalization(odds, slate):
    owls = parse_odds(odds, "nfl", slate, STAMP)
    legacy = parse_odds(odds, "nfl", slate, STAMP, provider="The Odds API")
    report = compare_boards(owls, legacy, slate, now=NOW)
    assert report["games"][0]["spread_difference"] == 0
    assert report["games"][0]["common_books"] == ["circa", "draftkings"]
    assert report["games"][0]["owls_freshness"] == "fresh"
    assert report["retirement_verified"] is False
    report = compare_boards({**owls, "games": []}, legacy, slate, now=NOW)
    assert report["missing_from_owls"] == [slate[0]["game_id"]]


def test_card_labels_separate_frozen_current_and_escape_sources(odds, slate):
    context = current_context(slate[0], parse_odds(odds, "nfl", slate, STAMP), now=NOW)
    context["reason"] = "<script>alert(1)</script>"
    rendered = market_context_html(slate[0], context)
    assert "Frozen projection" in rendered
    assert "Market at forecast: BUF -3.00" in rendered
    assert "Current market: BUF -3.75" in rendered
    assert "Current model edge (home): -1.75" in rendered
    assert "PUBLIC ACTION" in rendered
    assert "<script>" not in rendered


def test_collapsed_card_shows_averages_for_all_three_markets(odds, splits, slate):
    class CollapsedText(HTMLParser):
        def __init__(self):
            super().__init__()
            self.details = 0
            self.visible = []

        def handle_starttag(self, tag, attrs):
            if tag == "details":
                assert "open" not in dict(attrs)
                self.details += 1

        def handle_endtag(self, tag):
            if tag == "details":
                self.details -= 1

        def handle_data(self, data):
            if not self.details:
                self.visible.append(data)

    board = parse_odds(odds, "nfl", slate, STAMP)
    board["games"][0]["splits"] = parse_splits(splits)[EVENT]
    context = current_context(slate[0], board, now=NOW)
    context["splits"]["circa"]["stale"] = True
    parsed = CollapsedText()
    parsed.feed(market_context_html(slate[0], context))
    visible = " ".join(parsed.visible)
    for label in (
        "Average across 2 books",
        "Over / Under",
        "Over",
        "Under",
        "Spread",
        "Moneyline",
        "Tickets",
        "Handle",
        "BUF",
        "NYJ",
        "50.0%",
        "55.0%",
        "70.0%",
        "62.0%",
        "71.0%",
        "Includes stale data",
    ):
        assert label in visible
    for hidden in (
        "Frozen projection",
        "Market at forecast",
        "Movement",
        "edge",
        "2026-",
        "DraftKings",
        "Circa",
        "no model inputs",
    ):
        assert hidden not in visible


def test_ncaaf_odds_consensus_uses_same_median(odds):
    slate = [
        {
            "game_id": 123,
            "home_team": "Alabama",
            "away_team": "Georgia",
            "start_date": KICKOFF,
            "predicted_home_margin": 1.0,
        }
    ]
    payload = json.loads(
        json.dumps(odds)
        .replace("Buffalo Bills", "Alabama Crimson Tide")
        .replace("New York Jets", "Georgia Bulldogs")
        .replace("nfl:", "ncaaf:")
    )
    board = parse_odds(payload, "ncaaf", slate, STAMP)
    assert board["games"][0]["spread"]["home_spread"] == -3.75
    assert current_context(slate[0], board, now=NOW)["current_home_edge"] == -2.75


def test_timestamp_only_updates_dont_duplicate_but_returning_price_does(
    tmp_path, odds, splits, slate
):
    store = MarketStore(tmp_path / "market.db")
    client = Client(odds, splits)
    poll_market("nfl", slate, store=store, client=client, now=NOW)
    for minute in (5, 10, 15):
        now = NOW + timedelta(minutes=minute)
        book = client.odds["data"]["draftkings"][0]["bookmakers"][0]
        book["last_update"] = now.isoformat()
        spread = -5 if minute == 10 else -3.5
        book["markets"][0]["outcomes"][0]["point"] = spread
        book["markets"][0]["outcomes"][1]["point"] = -spread
        poll_market("nfl", slate, store=store, client=client, now=now)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM market_observations").fetchone()[0] == 4
        rows = db.execute(
            "SELECT spread FROM market_observations WHERE sportsbook='draftkings' ORDER BY id"
        ).fetchall()
        assert rows == [(-3.5,), (-5.0,), (-3.5,)]


def test_published_batch_bytes_stay_immutable_during_poll(tmp_path, odds, splits, slate):
    from nfl_prediction.ledger import PredictionLedger

    path = PredictionLedger(tmp_path / "forecasts").record_batch(
        slate, model_hash="test", data_cutoff=STAMP, prediction_season=2026
    )
    before = path.read_bytes()
    store = MarketStore(tmp_path / "market.db")
    poll_market("nfl", slate, store=store, client=Client(odds, splits), now=NOW)
    current_context(json.loads(before)["predictions"][0], store.read("nfl"), now=NOW)
    assert path.read_bytes() == before


def test_legacy_configuration_preserves_existing_context(monkeypatch, slate):
    monkeypatch.setenv("GRIDLINE_MARKET_PROVIDER", "legacy")
    assert freeze_market_context(slate, "nfl", as_of=NOW) == slate


def test_legacy_cfb_ledger_keeps_market_line_and_forecast_time(tmp_path):
    from nfl_prediction.ledger import PredictionLedger

    path = PredictionLedger(tmp_path).record_batch(
        [
            {
                "game_id": 123,
                "predicted_home_margin": 2.0,
                "market_consensus": {"spread": {"market_home_margin": 3.5}},
            }
        ],
        model_hash="test",
        data_cutoff=STAMP,
        prediction_season=2026,
    )
    batch = json.loads(path.read_text())
    assert batch["predictions"][0]["market_line"] == -3.5
    assert batch["predictions"][0]["forecast_at"] == batch["created_at"]


def test_provider_stale_and_missing_spread_are_not_current(odds, slate):
    odds["meta"]["freshness"]["stale"] = True
    board = parse_odds(odds, "nfl", slate, STAMP)
    assert current_context(slate[0], board, now=NOW)["status"] == "stale"
    odds["meta"]["freshness"] = False
    with pytest.raises(OwlsError, match="metadata"):
        parse_odds(odds, "nfl", slate, STAMP)


def test_quota_headers_pause_subsequent_requests(tmp_path, odds, splits, slate):
    store = MarketStore(tmp_path / "market.db")
    client = Client(odds, splits)
    client.quota_retry_after = 3600
    board = poll_market("nfl", slate, store=store, client=client, now=NOW)
    assert client.calls == [("nfl", "odds")]
    assert "quota" in board["splits_error"]
    poll_market("ncaaf", slate, store=store, client=client, now=NOW + timedelta(minutes=10))
    assert len(client.calls) == 1


@pytest.mark.parametrize("body", [b"not-json", b"[]", b'{"success":false}', b"\xff"])
def test_client_rejects_invalid_json_and_envelopes(monkeypatch, body):
    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return body

    monkeypatch.setenv("OWLS_INSIGHT_API_KEY", "test")
    monkeypatch.setattr("nfl_prediction.owls.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(OwlsError):
        OwlsClient().get("nfl", "odds")


def test_zero_splits_are_kept_and_malformed_splits_raise(splits):
    splits["data"][0]["splits"][0]["spread"].update(home_bets_pct=0, away_bets_pct=100)
    assert parse_splits(splits)[EVENT]["draftkings"]["markets"]["spread"]["home"]["ticket_pct"] == 0
    splits["data"][0]["splits"][0]["total"] = "invalid"
    with pytest.raises(OwlsError):
        parse_splits(splits)


def test_book_average_missing_pairs_and_sources(splits, slate):
    books = parse_splits(splits)[EVENT]
    original = copy.deepcopy(books)
    rendered = public_action_html(slate[0], {"splits": books})
    assert "50.0%" in rendered  # Spread tickets: equal weight for 40 and 60.
    assert "55.0%" in rendered  # Spread home handle: 45 and 65.
    assert "70.0%" in rendered  # Only DK has moneyline tickets.
    assert "62.0%" in rendered and "71.0%" in rendered  # O/U, one book.
    assert 'title="Circa, DraftKings"' in rendered
    assert "(0)</span>" in rendered  # No moneyline handle; never invent zero percent.
    assert books == original
    assert "Books disagree" in rendered
    books["draftkings"]["markets"]["moneyline"]["away"]["ticket_pct"] = None
    rendered = public_action_html(slate[0], {"splits": books})
    assert "70.0%" not in rendered  # A half-reported pair cannot skew the two sides.


def test_book_average_empty_and_zero_values(slate, splits):
    empty = public_action_html(slate[0], {})
    assert empty.count("Unavailable</div>") == 3
    assert "0.0%" not in empty
    books = parse_splits(splits)[EVENT]
    total = books["draftkings"]["markets"]["total"]
    total["over"]["ticket_pct"] = 0
    total["under"]["ticket_pct"] = 100
    rendered = public_action_html(slate[0], {"splits": books})
    assert ">0.0%</td>" in rendered and ">100.0%</td>" in rendered
    assert "Includes stale data" not in rendered


@pytest.mark.parametrize(
    "seconds,stale", [(900, False), (1800, False), (3600, False), (3601, True)]
)
def test_split_freshness_has_its_own_one_hour_window(odds, splits, slate, seconds, stale):
    board = parse_odds(odds, "nfl", slate, STAMP)
    books = parse_splits(splits)[EVENT]
    for book in books.values():
        book["source_timestamp"] = (NOW - timedelta(seconds=seconds)).isoformat()
    board["games"][0]["splits"] = books
    context = current_context(slate[0], board, now=NOW)
    assert context["splits"]["draftkings"]["stale"] is stale
    assert context["splits"]["draftkings"]["source_age_seconds"] == seconds
    assert context["status"] == "fresh"  # Odds use their own source timestamps.


def test_public_action_exposes_source_age_separately_from_poll_time(odds, splits, slate):
    board = parse_odds(odds, "nfl", slate, STAMP)
    books = parse_splits(splits)[EVENT]
    books["draftkings"]["source_timestamp"] = (NOW - timedelta(minutes=20)).isoformat()
    books["circa"]["source_timestamp"] = (NOW - timedelta(minutes=45)).isoformat()
    board["games"][0]["splits"] = books
    board["last_attempt_at"] = (NOW - timedelta(minutes=2)).isoformat()
    context = current_context(slate[0], board, now=NOW)
    rendered = public_action_html(slate[0], context)
    assert "Oldest source: 45m ago" in rendered
    assert rendered.count("Source: 20m ago") == 2  # Only DK contributes ML and totals.
    assert "Last check: 2m ago" in rendered
    assert "Includes stale data" not in rendered
    assert books["circa"]["source_timestamp"] in rendered
    board["splits_error"] = "Refresh failed"
    context = current_context(slate[0], board, now=NOW)
    rendered = public_action_html(slate[0], context)
    assert "Includes stale data" in rendered and "Refresh failed" in rendered
    assert "Oldest source: 45m ago" in rendered


@pytest.mark.parametrize("stamp", [None, "bad", (NOW + timedelta(minutes=1)).isoformat()])
def test_invalid_split_source_time_remains_explicit(odds, splits, slate, stamp):
    board = parse_odds(odds, "nfl", slate, STAMP)
    books = parse_splits(splits)[EVENT]
    books["circa"]["source_timestamp"] = stamp
    board["games"][0]["splits"] = books
    context = current_context(slate[0], board, now=NOW)
    assert context["splits"]["circa"]["stale"]
    assert "Oldest source: unavailable" in public_action_html(slate[0], context)
