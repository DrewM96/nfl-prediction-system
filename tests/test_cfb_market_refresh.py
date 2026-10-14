import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from cfb_prediction import production
from cfb_prediction.ledger import record_cfb_prediction_batch
from nfl_prediction import ledger
from nfl_prediction.current_market import MarketStore, poll_market
from nfl_prediction.odds import parse_timestamp
from nfl_prediction.owls import OwlsError
from nfl_prediction.results import forecast_rows, select_forecasts

NOW = datetime(2026, 10, 11, 13, tzinfo=UTC)
KICKOFF = (NOW + timedelta(days=2)).isoformat()


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW + timedelta(seconds=1)


class Client:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def get(self, sport, endpoint):
        self.calls.append((sport, endpoint))
        if self.fail:
            raise OwlsError("Provider unavailable", retry_after=300)
        if endpoint == "splits":
            return {"success": True, "data": {}}
        return {
            "success": True,
            "meta": {"timestamp": NOW.isoformat(), "freshness": {"stale": False}},
            "data": {
                "draftkings": [
                    {
                        "eventId": "ncaaf:Southern Miss@Troy-20261013",
                        "home_team": "Troy",
                        "away_team": "Southern Miss",
                        "commence_time": KICKOFF,
                        "bookmakers": [
                            {
                                "key": "draftkings",
                                "last_update": NOW.isoformat(),
                                "markets": [
                                    {
                                        "key": "spreads",
                                        "outcomes": [
                                            {"name": "Troy", "point": -10, "price": -110},
                                            {"name": "Southern Miss", "point": 10, "price": -110},
                                        ],
                                    },
                                    {
                                        "key": "totals",
                                        "outcomes": [
                                            {"name": "Over", "point": 54, "price": -110},
                                            {"name": "Under", "point": 54, "price": -110},
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ]
            },
        }


@pytest.fixture
def capture_setup(tmp_path, monkeypatch):
    store = MarketStore(tmp_path / "market.sqlite3")
    client = Client()
    monkeypatch.setenv("GRIDLINE_MARKET_PROVIDER", "owls")
    monkeypatch.setattr(production, "MarketStore", lambda: store)
    monkeypatch.setattr(production, "datetime", Clock)
    monkeypatch.setattr(ledger, "datetime", Clock)
    slates = []

    def poll(sport, slate, *, store):
        slates.append(deepcopy(slate))
        return poll_market(sport, slate, store=store, client=client, now=NOW)

    monkeypatch.setattr(production, "poll_market", poll)
    prediction = dict(
        game_id=401871090,
        season=2026,
        week=7,
        home_team="Troy",
        away_team="Southern Miss",
        start_date=KICKOFF,
        neutral_site=False,
        predicted_home_margin=12.3,
        predicted_total=55,
    )
    return store, client, slates, [prediction]


def test_selected_upcoming_slate_is_polled_and_lines_survive_publication(capture_setup, tmp_path):
    store, client, slates, predictions = capture_setup
    original = deepcopy(predictions)
    cutoff = NOW - timedelta(minutes=30)
    # The prior release can point at completed games; it must not select the poll slate.
    pointer = tmp_path / "latest_prediction.json"
    pointer.write_text(json.dumps({"path": "old-week.json"}), encoding="utf-8")
    frozen, summary = production._capture_market_context(predictions, as_of=cutoff, refresh=True)
    assert predictions == original
    assert slates == [original]
    assert client.calls == [("ncaaf", "odds"), ("ncaaf", "splits")]
    assert frozen[0]["market_consensus"]["spread"]["home_spread"] == -10
    assert frozen[0]["market_consensus"]["total"]["total"] == 54
    assert summary["games_with_spreads"] == summary["games_with_totals"] == 1
    assert summary["odds_error"] is None
    assert cutoff < parse_timestamp(frozen[0]["market_consensus"]["snapshot_at"])
    assert frozen[0]["forecast_at"] == Clock.now().isoformat()
    path = record_cfb_prediction_batch(
        frozen,
        model_hash="original-model",
        data_cutoff=cutoff.isoformat(),
        prediction_season=2026,
        metadata={"market_refresh": summary},
        root=tmp_path / "predictions",
        latest_path=pointer,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["data_cutoff"] == cutoff.isoformat()
    assert payload["predictions"][0]["predicted_home_margin"] == 12.3
    assert parse_timestamp(payload["created_at"]) >= parse_timestamp(
        frozen[0]["market_consensus"]["snapshot_at"]
    )
    selected = select_forecasts(
        forecast_rows(path.parent, as_of=Clock.now()), policy="horizon"
    ).iloc[0]
    assert selected.market_margin == 10
    assert selected.market_total == 54


def test_cached_or_historical_capture_does_not_poll(capture_setup):
    store, client, slates, predictions = capture_setup
    frozen, summary = production._capture_market_context(predictions, as_of=NOW, refresh=False)
    assert not summary["attempted"]
    assert client.calls == slates == []
    assert frozen[0]["forecast_at"] == NOW.isoformat()


def test_legacy_provider_does_not_poll_or_replace_existing_lines(capture_setup, monkeypatch):
    store, client, slates, predictions = capture_setup
    monkeypatch.setenv("GRIDLINE_MARKET_PROVIDER", "legacy")
    predictions[0]["market_consensus"] = {"provider": "legacy", "spread": {"home_spread": -7}}
    frozen, summary = production._capture_market_context(predictions, as_of=NOW, refresh=True)
    assert frozen == predictions
    assert not summary["attempted"]
    assert client.calls == []


def test_failed_provider_is_reported_without_inventing_lines(capture_setup, caplog):
    store, client, slates, predictions = capture_setup
    client.fail = True
    frozen, summary = production._capture_market_context(predictions, as_of=NOW, refresh=True)
    assert summary["odds_error"] == "Provider unavailable"
    assert summary["games_with_spreads"] == summary["games_with_totals"] == 0
    assert frozen[0]["market_consensus"] is None
    assert frozen[0]["predicted_home_margin"] == 12.3
    assert "No eligible spreads" in caplog.text


def test_refresh_preserves_provider_poll_cadence(capture_setup):
    store, client, slates, predictions = capture_setup
    production._capture_market_context(predictions, as_of=NOW, refresh=True)
    production._capture_market_context(predictions, as_of=NOW, refresh=True)
    assert client.calls == [("ncaaf", "odds"), ("ncaaf", "splits")]


@pytest.mark.parametrize("historical", [False, True])
def test_production_selects_week_before_market_poll_and_skips_historical_requests(
    capture_setup,
    tmp_path,
    monkeypatch,
    historical,
):
    store, client, slates, predictions = capture_setup
    features = pd.DataFrame(
        [
            dict(predictions[0], home_field=True, completed=False, home_margin=0),
            dict(
                predictions[0],
                game_id=2,
                week=8,
                start_date=NOW + timedelta(days=9),
                home_field=True,
                completed=False,
                home_margin=0,
            ),
            dict(
                predictions[0],
                game_id=3,
                season=2025,
                week=7,
                start_date=NOW - timedelta(days=365),
                home_field=True,
                completed=True,
                home_margin=7,
            ),
        ]
    )
    features["start_date"] = pd.to_datetime(features.start_date, utc=True)
    events = []

    def model(*args, name, **kwargs):
        events.append(name)
        mu = 12.3 if name == "margin" else 55
        return SimpleNamespace(
            estimator=object(),
            distribution=lambda rows: [
                {"mean": mu, "probability_above_zero": 0.6, "p10": mu - 10, "p90": mu + 10}
                for _ in range(len(rows))
            ],
        )

    def save_bundle(*args, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {"benchmark_sha256": "benchmark"}

    original_poll = production.poll_market

    def poll(*args, **kwargs):
        assert events == ["margin", "total"]
        return original_poll(*args, **kwargs)

    monkeypatch.setattr(production, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(production, "load_historical_data", lambda *a, **kw: object())
    monkeypatch.setattr(production, "_combine", lambda parts: SimpleNamespace(games=pd.DataFrame()))
    monkeypatch.setattr(production, "_cut_off_results", lambda data, cutoff: data)
    monkeypatch.setattr(production, "build_point_in_time_features", lambda *a, **kw: features)
    monkeypatch.setattr(production, "_current_input_coverage", lambda *a: {})
    monkeypatch.setattr(production, "CFB_FEATURE_CONFIGURATIONS", {"test": []})
    monkeypatch.setattr(
        production,
        "_load_selected_benchmark",
        lambda path: (
            {"min_train_rows": 1, "ridge_alpha": 1},
            {"margin": "test", "total": "test"},
        ),
    )
    monkeypatch.setattr(production, "fit_cfb_model", model)
    monkeypatch.setattr(production, "save_cfb_model_bundle", save_bundle)
    monkeypatch.setattr(
        production, "make_ridge", lambda alpha: SimpleNamespace(fit=lambda *a: object())
    )
    monkeypatch.setattr(
        production, "build_blended_cfb_power_ratings", lambda *a, **kw: {"display_count": 0}
    )
    monkeypatch.setattr(production, "poll_market", poll)
    result = production.run_cfb_production_update(
        prediction_season=2026,
        historical_seasons=[2025],
        client=object(),
        as_of=NOW if historical else None,
        refresh_current=True,
        models_dir=tmp_path / "models",
        predictions_dir=tmp_path / "predictions",
        latest_path=tmp_path / "latest_prediction.json",
    )
    assert result["forecast_week"] == 7
    assert result["prediction_count"] == 1  # Week 8 is excluded from this publication and poll.
    if historical:
        assert not slates and not client.calls
    else:
        assert len(slates) == 1 and [p["week"] for p in slates[0]] == [7]
        assert result["predictions"][0]["market_consensus"]["total"]["total"] == 54
