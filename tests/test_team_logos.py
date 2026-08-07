from __future__ import annotations

import json
from pathlib import Path

from nfl_prediction.ui import TEAM_NAMES
from team_logos import team_logo_url


def test_logo_registry_covers_current_nfl_and_fbs_teams() -> None:
    registry = json.loads(Path("data/team_logos.json").read_text(encoding="utf-8"))
    rankings = json.loads(Path("data/cfb/power_rankings.json").read_text(encoding="utf-8"))

    assert set(registry["nfl"]) == set(TEAM_NAMES)
    assert len(registry["cfb"]) == 138
    assert not {row["team"] for row in rankings["ratings"]} - set(registry["cfb"])
    assert all(url.startswith("https://") for url in registry["nfl"].values())
    assert all(url.startswith("https://") for url in registry["cfb"].values())


def test_logo_lookup_supports_current_and_provider_nfl_abbreviations() -> None:
    assert team_logo_url("LA", "nfl").endswith("/lar.png")
    assert team_logo_url("LAR", "nfl").endswith("/lar.png")
    assert team_logo_url("WAS", "nfl").endswith("/wsh.png")
    assert team_logo_url("WSH", "nfl").endswith("/wsh.png")
    assert team_logo_url("LSU", "cfb").endswith("/99.png")
    assert team_logo_url("Unknown", "cfb") is None
