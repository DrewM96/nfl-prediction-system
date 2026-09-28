from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_prediction.config import PROJECT_ROOT, get_season_context
from nfl_prediction.io import read_json
from nfl_prediction.weather import capture_weather


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive upcoming NFL weather for prospective research"
    )
    parser.add_argument("--schedule", type=Path)
    args = parser.parse_args()
    if args.schedule:
        schedules = pd.read_parquet(args.schedule)
    else:
        import nflreadpy as nfl

        schedules = nfl.load_schedules([get_season_context().prediction_season]).to_pandas()
    schedules = schedules.astype(object).where(pd.notna(schedules), None)
    path = capture_weather(
        schedules.to_dict("records"),
        read_json(PROJECT_ROOT / "data/weather_venues.json"),
        PROJECT_ROOT / "data/weather",
        PROJECT_ROOT / "data/weather_private",
    )
    print(path)


if __name__ == "__main__":
    main()
