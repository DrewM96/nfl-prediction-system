from datetime import UTC, datetime

import market_update
from nfl_prediction import pipeline
from nfl_prediction.owls import OwlsError

NOW = datetime(2026, 10, 6, 12, tzinfo=UTC)


def test_owls_cli_publishes_without_legacy_client(monkeypatch, capsys):
    def legacy(*args, **kwargs):
        raise AssertionError("The cancelled provider must not be called")

    monkeypatch.setattr(market_update, "OddsApiClient", legacy)
    monkeypatch.setattr(
        "nfl_prediction.market_snapshot.refresh_owls_consensus",
        lambda: {
            "provider": "Owls Insight",
            "snapshot_at": NOW.isoformat(),
            "games": [{}],
        },
    )
    assert market_update.main(["--provider", "owls"]) == 0
    assert "Owls Insight" in capsys.readouterr().out


def test_owls_cli_failure_is_nonzero(monkeypatch):
    def fail():
        raise OwlsError("Owls authentication failed")

    monkeypatch.setattr("nfl_prediction.market_snapshot.refresh_owls_consensus", fail)
    assert market_update.main(["--provider", "owls"]) == 1


def test_forecast_refreshes_new_slate_without_legacy_fallback(monkeypatch):
    monkeypatch.setenv("GRIDLINE_MARKET_PROVIDER", "owls")
    slate = [{"game_id": "new-week"}]
    snapshot = {"provider": "Owls Insight", "games": slate}
    calls = []

    def refresh(predictions, *, as_of):
        calls.append((predictions, as_of))
        return snapshot

    def legacy():
        raise AssertionError("No legacy fallback")

    monkeypatch.setattr(pipeline, "refresh_owls_consensus", refresh)
    monkeypatch.setattr(pipeline, "load_market_consensus", legacy)
    assert pipeline._forecast_market_snapshot(slate, as_of=NOW) == snapshot
    assert calls == [(slate, NOW)]

    def fail(*args, **kwargs):
        raise OwlsError("Owls unavailable")

    monkeypatch.setattr(pipeline, "refresh_owls_consensus", fail)
    assert pipeline._forecast_market_snapshot(slate, as_of=NOW) is None
