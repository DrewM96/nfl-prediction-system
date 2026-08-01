from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cfb_prediction.pipeline import build_foundation_summary, load_foundation_data


class FakeClient:
    def get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any],
        refresh: bool,
    ) -> list[dict[str, Any]]:
        assert params["year"] == 2026
        assert refresh is False
        if endpoint == "/teams/fbs":
            return [{"id": 1, "school": "Alpha", "classification": "fbs"}]
        if endpoint == "/games":
            return [
                {
                    "id": 2,
                    "season": 2026,
                    "week": 1,
                    "homeClassification": "fbs",
                    "awayClassification": "fbs",
                    "completed": False,
                }
            ]
        return [
            {
                "season": 2026,
                "week": 1,
                "seasonType": "regular",
                "startDate": "2026-08-29T07:00:00Z",
                "endDate": "2026-09-08T06:59:00Z",
            },
            {
                "season": 2026,
                "week": 1,
                "seasonType": "postseason",
                "startDate": "2026-12-12T08:00:00Z",
                "endDate": "2027-01-28T07:59:00Z",
            },
        ]


def test_foundation_summary_publishes_aggregates_not_raw_records(tmp_path: Path) -> None:
    data = load_foundation_data(FakeClient(), 2026)
    summary = build_foundation_summary(
        data,
        2026,
        generated_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    path = tmp_path / "foundation.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    assert summary["team_count"] == 1
    assert summary["fbs_vs_fbs_game_count"] == 1
    assert summary["calendar_week_count"] == 1
    assert summary["season_end"].startswith("2026-09-08")
    assert summary["raw_data_published"] is False
    assert "Alpha" not in path.read_text(encoding="utf-8")
