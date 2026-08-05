from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfb_prediction.rankings import build_cfb_power_ratings


def test_cfb_rankings_recover_team_strength_and_home_field() -> None:
    strengths = {"Alpha": 8.0, "Bravo": 3.0, "Charlie": -2.0, "Delta": -9.0}
    games = []
    margins = []
    for first, second in itertools.combinations(strengths, 2):
        for home, away, home_field in ((first, second, 1.0), (second, first, 0.0)):
            games.append({"home_team": home, "away_team": away, "home_field": home_field})
            margins.append(strengths[home] - strengths[away] + 2.5 * home_field + 0.2)

    result = build_cfb_power_ratings(
        pd.DataFrame(games),
        np.asarray(margins),
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        prediction_season=2026,
        data_cutoff="2026-08-01T00:00:00+00:00",
        model_hash="abc123",
        input_coverage={"scheduled_fbs_teams": 4},
        display_count=3,
        ridge_strength=0.0,
    )

    assert result["display_count"] == 3
    assert result["home_field_points"] == pytest.approx(2.7)
    assert result["incremental_home_field_points"] == pytest.approx(2.5)
    assert result["designated_home_bias"] == pytest.approx(0.2)
    assert result["line_fit_mae"] == pytest.approx(0.0, abs=1e-10)
    ratings = {row["team"]: row["rating"] for row in result["ratings"]}
    assert ratings == pytest.approx(strengths)


def test_published_cfb_top_30_matches_active_model() -> None:
    rankings = json.loads(Path("data/cfb/power_rankings.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path("data/cfb/models/manifest.json").read_text(encoding="utf-8"))

    assert rankings["model_hash"]
    assert rankings["prediction_season"] == manifest["prediction_season"] == 2026
    assert rankings["display_count"] == 30
    assert rankings["team_count"] == 138
    assert rankings["game_count"] >= 700
    assert len(rankings["ratings"]) == 138
    assert [row["rank"] for row in rankings["ratings"]] == list(range(1, 139))
    assert rankings["line_fit_mae"] < 3.0
