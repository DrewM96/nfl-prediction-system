from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nfl_prediction.io import read_json

LOGO_REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "team_logos.json"
NFL_ALIASES = {"LAR": "LA", "WSH": "WAS"}


@lru_cache(maxsize=1)
def load_team_logo_registry() -> dict[str, dict[str, str]]:
    registry = read_json(LOGO_REGISTRY_PATH, {})
    return {
        "nfl": dict(registry.get("nfl", {})),
        "cfb": dict(registry.get("cfb", {})),
    }


def team_logo_url(team: str, sport: str) -> str | None:
    league = sport.strip().lower()
    key = str(team).strip()
    if league == "nfl":
        key = NFL_ALIASES.get(key, key)
    return load_team_logo_registry().get(league, {}).get(key)
