from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODEL_MANIFEST_PATH = MODELS_DIR / "manifest.json"


@dataclass(frozen=True)
class SeasonContext:
    prediction_season: int
    training_seasons: tuple[int, ...]
    as_of: date


def get_season_context(
    as_of: date | datetime | None = None,
    *,
    history_seasons: int = 4,
) -> SeasonContext:
    """Return explicit training and prediction seasons.

    The NFL league year turns over in mid-March. Separating the prediction
    season from completed training seasons lets the system load the upcoming
    schedule during spring/summer without pretending future games exist.
    """
    current = as_of.date() if isinstance(as_of, datetime) else (as_of or date.today())
    prediction_season = (
        current.year if (current.month, current.day) >= (3, 15) else current.year - 1
    )
    first_training = prediction_season - history_seasons
    training_seasons = tuple(range(first_training, prediction_season + 1))
    return SeasonContext(prediction_season, training_seasons, current)


DIVISIONS: dict[str, frozenset[str]] = {
    "AFC East": frozenset({"BUF", "MIA", "NE", "NYJ"}),
    "AFC North": frozenset({"BAL", "CIN", "CLE", "PIT"}),
    "AFC South": frozenset({"HOU", "IND", "JAX", "TEN"}),
    "AFC West": frozenset({"DEN", "KC", "LV", "LAC"}),
    "NFC East": frozenset({"DAL", "NYG", "PHI", "WAS"}),
    "NFC North": frozenset({"CHI", "DET", "GB", "MIN"}),
    "NFC South": frozenset({"ATL", "CAR", "NO", "TB"}),
    "NFC West": frozenset({"ARI", "LA", "SF", "SEA"}),
}


def is_division_game(team_a: str, team_b: str) -> bool:
    return any(team_a in teams and team_b in teams for teams in DIVISIONS.values())
