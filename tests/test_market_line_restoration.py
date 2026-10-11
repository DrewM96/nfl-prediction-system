import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from nfl_prediction.results import _restore_market_lines, forecast_rows, select_forecasts


def _row(run_id, published_at, *, margin=None, total=None, captured=None):
    return dict(
        season=2026,
        game_id="401871090",
        home_team="Troy",
        away_team="Southern Miss",
        neutral_site=False,
        kickoff=datetime(2026, 10, 7, tzinfo=UTC),
        eligible=True,
        run_id=run_id,
        published_at=published_at,
        market_margin=margin,
        market_total=total,
        market_at=captured,
        market_source_at=None,
        published_margin=12.3,
        actual_margin=21,
    )


def _releases():
    first = _row(
        "earlier",
        datetime(2026, 10, 4, 16, tzinfo=UTC),
        margin=10,
        total=54,
        captured="2026-10-04T15:46:08+00:00",
    )
    latest = _row("selected", datetime(2026, 10, 5, 20, tzinfo=UTC))
    return first, latest


def test_restores_missing_lines_with_original_projections_and_provenance():
    earlier, latest = _releases()
    original = deepcopy(latest)
    rows = _restore_market_lines([latest, earlier])  # Input order does not choose the source.
    assert rows[0]["market_margin"] == 10
    assert rows[0]["market_total"] == 54
    assert rows[0]["embedded_market_margin"] is None
    assert rows[0]["market_margin_run_id"] == "earlier"
    assert rows[0]["market_margin_at"] == earlier["market_at"]
    assert rows[0]["market_margin_recovered"]
    assert all(
        rows[0][key] == value
        for key, value in original.items()
        if key not in {"market_margin", "market_total"}
    )


def test_existing_selected_line_wins_and_each_market_recovers_independently():
    earlier, latest = _releases()
    latest.update(market_margin=11, market_at="2026-10-05T19:59:00+00:00")
    _restore_market_lines([earlier, latest])
    assert latest["market_margin"] == 11
    assert not latest["market_margin_recovered"]
    assert latest["market_margin_run_id"] == "selected"
    assert latest["market_total"] == 54
    assert latest["market_total_run_id"] == "earlier"


@pytest.mark.parametrize(
    "changes",
    [
        {"home_team": "Southern Miss", "away_team": "Troy"},
        {"game_id": "different"},
        {"season": 2025},
        {"neutral_site": True},
        {"kickoff": datetime(2026, 10, 8, tzinfo=UTC)},
        {"eligible": False},
        {"published_at": datetime(2026, 10, 6, tzinfo=UTC)},
        {"market_at": "2026-10-04T16:01:00+00:00"},
        {"market_source_at": "2026-10-04T16:00:00+00:00"},
        {"market_source_at": "invalid"},
        {"market_margin": float("nan"), "market_total": float("inf")},
    ],
)
def test_rejects_mismatched_or_unverified_sources(changes):
    earlier, latest = _releases()
    earlier.update(changes)
    _restore_market_lines([earlier, latest])
    assert latest["market_margin"] is None
    assert latest["market_total"] is None


def test_source_expires_under_selected_forecast_eight_day_limit():
    earlier, latest = _releases()
    earlier["market_at"] = (latest["published_at"] - timedelta(days=8, seconds=1)).isoformat()
    _restore_market_lines([earlier, latest])
    assert latest["market_margin"] is None


def test_latest_capture_wins_without_using_outcomes():
    earlier, latest = _releases()
    newer = deepcopy(earlier)
    newer.update(
        run_id="newer",
        published_at=datetime(2026, 10, 5, 12, tzinfo=UTC),
        market_at="2026-10-05T11:59:00+00:00",
        market_margin=11,
    )
    _restore_market_lines([latest, newer, earlier])
    assert latest["market_margin"] == 11
    assert latest["market_margin_run_id"] == "newer"


def test_forecast_reader_restores_without_rewriting_archives(tmp_path):
    earlier, latest = _releases()
    originals = {}
    for row in (earlier, latest):
        prediction = dict(
            game_id=row["game_id"],
            season=row["season"],
            week=6,
            home_team=row["home_team"],
            away_team=row["away_team"],
            start_date=row["kickoff"].isoformat(),
            neutral_site=False,
            predicted_home_margin=12.3,
            predicted_total=55,
        )
        if row["market_at"]:
            prediction["market_consensus"] = dict(
                snapshot_at=row["market_at"],
                spread={"home_spread": -10},
                total={"total": 54},
            )
        batch = dict(
            run_id=row["run_id"],
            created_at=row["published_at"].isoformat(),
            prediction_season=2026,
            data_cutoff="2026-10-04",
            model_hash="frozen",
            predictions=[prediction],
        )
        path = tmp_path / f"{row['run_id']}.json"
        path.write_text(json.dumps(batch), encoding="utf-8")
        originals[path] = path.read_bytes()
    versions = forecast_rows(tmp_path, as_of=datetime(2026, 10, 11, tzinfo=UTC))
    selected = select_forecasts(versions, policy="horizon").iloc[0]
    assert selected.run_id == "selected"
    assert selected.published_margin == 12.3
    assert selected.market_margin == 10
    assert selected.market_margin_recovered
    assert not versions[versions.run_id.eq("earlier")].iloc[0].market_margin_recovered
    assert all(path.read_bytes() == content for path, content in originals.items())
