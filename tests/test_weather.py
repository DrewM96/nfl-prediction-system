from datetime import UTC, datetime

import pytest

from nfl_prediction.io import read_json
from nfl_prediction.weather import capture_weather, summarize_weather


def test_weather_requires_complete_game_window():
    kickoff = datetime(2026, 10, 1, 17, tzinfo=UTC)
    with pytest.raises(ValueError, match="cover"):
        summarize_weather({"hourly": {"time": ["2026-10-01T17:00"]}}, kickoff)
    hourly = {"time": [f"2026-10-01T{hour}:00" for hour in range(17, 21)]}
    for key in ("temperature_2m", "wind_speed_10m", "wind_gusts_10m", "precipitation"):
        hourly[key] = [1, 2, 3, 4]
    assert summarize_weather({"hourly": hourly}, kickoff)["precipitation_mm_sum"] == 10
    hourly["wind_gusts_10m"][0] = None
    with pytest.raises(ValueError, match="wind_gusts"):
        summarize_weather({"hourly": hourly}, kickoff)


def test_venue_mismatch_does_not_fetch_weather_for_wrong_city(tmp_path):
    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError("Wrong venue must not be queried")

    games = [
        dict(
            game_id="one",
            stadium_id="JAX00",
            stadium="London Stadium",
            gameday="2026-10-01",
            gametime="13:00",
        )
    ]
    venues = {"JAX00": {"names": ["EverBank Stadium"]}}
    path = capture_weather(
        games,
        venues,
        tmp_path,
        tmp_path / "raw",
        session=NoNetwork(),
        now=datetime(2026, 9, 28, tzinfo=UTC),
    )
    assert read_json(path)["games"][0]["reason"] == "unverified_venue_name_or_coordinates"
