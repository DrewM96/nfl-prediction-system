from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError

import pytest

from nfl_prediction.odds import (
    CreditUsage,
    MarketSnapshotStore,
    OddsApiClient,
    OddsApiError,
    OddsFetchResult,
    attach_market_consensus,
    build_consensus,
    estimate_historical_credits,
)


def _fetch() -> OddsFetchResult:
    def book(key: str, spread: float, total: float, price: float) -> dict:
        return {
            "key": key,
            "last_update": "2026-09-10T15:55:00Z",
            "markets": [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Buffalo Bills", "point": spread, "price": price},
                        {"name": "New York Jets", "point": -spread, "price": -110},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": total, "price": -108},
                        {"name": "Under", "point": total, "price": -112},
                    ],
                },
            ],
        }

    return OddsFetchResult(
        payload={
            "timestamp": "2026-09-10T16:00:00Z",
            "data": [
                {
                    "home_team": "Buffalo Bills",
                    "away_team": "New York Jets",
                    "commence_time": "2026-09-13T17:00:00Z",
                    "bookmakers": [book("one", -3.5, 46.0, -105), book("two", -4.0, 47.0, -115)],
                }
            ],
        },
        captured_at="2026-09-10T16:01:00+00:00",
        credits=CreditUsage(remaining=980, used=20, last_request=20),
        request_kind="historical",
    )


def test_consensus_normalizes_team_names_spread_sign_prices_and_dispersion() -> None:
    consensus = build_consensus(_fetch(), regions="us", markets=("spreads", "totals"))
    game = consensus["games"][0]
    assert (game["away_team"], game["home_team"]) == ("NYJ", "BUF")
    assert game["spread"]["home_spread"] == -3.75
    assert game["spread"]["market_home_margin"] == 3.75
    assert game["spread"]["home_price"] == -110
    assert game["spread"]["book_count"] == 2
    assert game["spread"]["line_iqr"] == 0.25
    assert game["spread"]["latest_book_update"] == "2026-09-10T15:55:00+00:00"
    assert game["total"]["total"] == 46.5
    assert "bookmakers" not in game


def test_attach_market_keeps_independent_forecast_and_rejects_future_knowledge() -> None:
    consensus = build_consensus(_fetch(), regions="us", markets=("spreads", "totals"))
    independent = [{"away_team": "NYJ", "home_team": "BUF", "predicted_home_margin": 1.0}]
    accepted = attach_market_consensus(
        independent,
        consensus,
        as_of=datetime(2026, 9, 10, 17, tzinfo=UTC),
    )[0]
    assert accepted["predicted_home_margin"] == 1.0
    assert accepted["market_informed"]["home_margin"] == 3.75
    assert accepted["market_line"] == -3.75

    rejected = attach_market_consensus(
        independent,
        consensus,
        as_of=datetime(2026, 9, 10, 15, tzinfo=UTC),
    )[0]
    assert rejected["market_consensus"] is None
    assert rejected["market_informed"] is None


def test_attach_market_rejects_stale_or_post_kickoff_snapshots() -> None:
    consensus = build_consensus(_fetch(), regions="us", markets=("spreads", "totals"))
    prediction = [{"away_team": "NYJ", "home_team": "BUF"}]
    stale = attach_market_consensus(
        prediction,
        consensus,
        max_age=timedelta(days=1),
        as_of=datetime(2026, 9, 13, 16, tzinfo=UTC),
    )[0]
    assert stale["market_consensus"] is None

    consensus["snapshot_at"] = "2026-09-13T18:00:00+00:00"
    after_kickoff = attach_market_consensus(
        prediction,
        consensus,
        as_of=datetime(2026, 9, 13, 19, tzinfo=UTC),
    )[0]
    assert after_kickoff["market_consensus"] is None


def test_private_store_is_immutable_and_consensus_is_sanitized(tmp_path) -> None:
    fetch = _fetch()
    consensus = build_consensus(fetch, regions="us", markets=("spreads", "totals"))
    store = MarketSnapshotStore(tmp_path / "private", tmp_path / "consensus.json")
    raw_path, consensus_path = store.save(fetch, consensus)
    assert raw_path.exists()
    assert consensus_path.exists()
    assert "bookmakers" not in consensus_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        store.save(fetch, consensus)


def test_historical_credit_estimate_scales_by_market_region_and_snapshot() -> None:
    assert estimate_historical_credits(markets=("spreads", "totals")) == 20
    assert estimate_historical_credits(regions="us,uk", snapshots=3) == 120


def test_client_returns_quota_without_exposing_key(monkeypatch) -> None:
    class Response:
        headers = {"x-requests-remaining": "499", "x-requests-used": "1", "x-requests-last": "1"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b"[]"

    monkeypatch.setattr("nfl_prediction.odds.urlopen", lambda *_args, **_kwargs: Response())
    result = OddsApiClient("do-not-print-me").current_odds(markets=("spreads",))
    assert result.credits.remaining == 499
    assert "do-not-print-me" not in repr(result)


def test_http_error_is_redacted(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise HTTPError(
            "https://example.test?apiKey=do-not-print-me", 401, "Unauthorized", {}, None
        )

    monkeypatch.setattr("nfl_prediction.odds.urlopen", fail)
    with pytest.raises(OddsApiError) as caught:
        OddsApiClient("do-not-print-me").current_odds()
    assert "do-not-print-me" not in str(caught.value)
    assert caught.value.__cause__ is None
