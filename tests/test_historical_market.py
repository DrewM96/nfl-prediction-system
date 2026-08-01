from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from nfl_prediction.historical_market import (
    SnapshotRequest,
    add_prequential_market_blends,
    build_aggregate_report,
    build_snapshot_plan,
    collect_historical_consensus,
    learn_mae_weight,
    prequential_component_blend,
)
from nfl_prediction.io import atomic_write_json


def test_snapshot_plan_groups_one_request_per_kickoff_window() -> None:
    games = pd.DataFrame(
        [
            {"gameday": "2025-09-07", "gametime": "13:00"},
            {"gameday": "2025-09-07", "gametime": "13:00"},
            {"gameday": "2025-09-07", "gametime": "16:25"},
        ]
    )
    plan = build_snapshot_plan(games, minutes_before_kickoff=30)
    assert len(plan) == 2
    assert plan[0].requested_at == "2025-09-07T16:30:00+00:00"
    assert plan[0].kickoff_at == "2025-09-07T17:00:00+00:00"
    assert plan[0].game_indices == (0, 1)


def test_mae_weight_prefers_the_better_component() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    assert learn_mae_weight(actual, actual, np.zeros(3)) == 1.0
    assert learn_mae_weight(actual, np.zeros(3), actual) == 0.0


def test_prequential_component_weights_cannot_see_future_weeks() -> None:
    validation = pd.DataFrame({"season": [2025] * 6, "week": [1, 1, 2, 2, 3, 3]})
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    first = actual.copy()
    second = np.zeros(6)
    predictions, weights = prequential_component_blend(
        validation, actual, first, second, min_meta_rows=2
    )
    altered = actual.copy()
    altered[4:] = 1000.0
    altered_predictions, altered_weights = prequential_component_blend(
        validation, altered, first, second, min_meta_rows=2
    )
    assert predictions[:4] == pytest.approx(altered_predictions[:4])
    assert weights[:4] == pytest.approx(altered_weights[:4])
    assert weights[2] == 1.0


def _benchmark_frame() -> pd.DataFrame:
    rows = []
    for week in range(1, 5):
        rows.append(
            {
                "season": 2025,
                "week": week,
                "gameday": f"2025-09-{week + 6:02d}",
                "gametime": "13:00",
                "actual_home_margin": float(week),
                "actual_total_points": 40.0 + week,
                "independent_home_margin": float(week + 2),
                "independent_total": 42.0 + week,
                "market_home_margin": float(week),
                "market_total": 40.0 + week,
                "spread_book_count": 8,
                "total_book_count": 7,
            }
        )
    return pd.DataFrame(rows)


def test_market_blend_uses_prior_rows_only() -> None:
    frame = _benchmark_frame()
    blended = add_prequential_market_blends(frame, min_meta_rows=2)
    assert blended.loc[0, "market_margin_weight"] == 0.5
    assert blended.loc[2, "market_margin_weight"] == 1.0


def test_aggregate_report_contains_metrics_not_game_records() -> None:
    report = build_aggregate_report(
        _benchmark_frame(),
        {"request_count": 4, "estimated_credits": 80, "matched_games": 4},
        training_seasons=[2022, 2023, 2024, 2025],
        evaluation_seasons=[2025],
        minutes_before_kickoff=30,
    )
    assert report["games"] == 4
    assert report["variants"]["market_margin"]["mae"] == 0.0
    assert "game_id" not in report
    assert report["methodology"]["raw_market_data_published"] is False
    assert datetime.fromisoformat(report["generated_at"]).tzinfo == UTC


def test_cached_snapshot_uses_zero_api_credits(tmp_path) -> None:
    request = SnapshotRequest(
        requested_at="2025-09-07T16:30:00+00:00",
        kickoff_at="2025-09-07T17:00:00+00:00",
        game_indices=(0,),
    )
    consensus_path = tmp_path / "consensus" / "benchmark-20250907T163000Z.json"
    atomic_write_json(
        consensus_path,
        {
            "snapshot_at": "2025-09-07T16:30:00+00:00",
            "games": [
                {
                    "away_team": "NYJ",
                    "home_team": "BUF",
                    "commence_time": "2025-09-07T17:00:00+00:00",
                    "spread": {
                        "home_spread": -3.0,
                        "market_home_margin": 3.0,
                        "book_count": 8,
                        "line_iqr": 0.5,
                    },
                    "total": {"total": 44.5, "book_count": 8, "line_iqr": 0.5},
                }
            ],
        },
    )

    class Client:
        def historical_odds(self, *_args, **_kwargs):
            raise AssertionError("cached request must not call the API")

    games = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "gameday": "2025-09-07",
                "gametime": "13:00",
                "away_team": "NYJ",
                "home_team": "BUF",
            }
        ]
    )
    collected, metadata = collect_historical_consensus(
        games,
        [request],
        client=Client(),  # type: ignore[arg-type]
        max_credits=0,
        private_root=tmp_path,
    )
    assert len(collected) == 1
    assert metadata["api_requests"] == 0
    assert metadata["cache_hits"] == 1
    assert metadata["estimated_credits"] == 0
    assert metadata["actual_credits"] == 0
