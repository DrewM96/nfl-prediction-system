"""Timestamped weather observations for research; no production point adjustments."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import requests

from .io import atomic_write_json
from .odds import _game_kickoff

VARIABLES = ("temperature_2m", "wind_speed_10m", "wind_gusts_10m", "precipitation")


def summarize_weather(payload: dict, kickoff: datetime) -> dict:
    hourly = payload.get("hourly") or {}
    if payload.get("utc_offset_seconds", 0) != 0:
        raise ValueError("Expected UTC weather response")
    times = [datetime.fromisoformat(value).replace(tzinfo=UTC) for value in hourly.get("time", [])]
    start = kickoff.replace(minute=0, second=0, microsecond=0)
    indices = [i for i, time in enumerate(times) if start <= time < kickoff + timedelta(hours=4)]
    expected = {start + timedelta(hours=i) for i in range(5 if kickoff.minute else 4)}
    if len(indices) != len(expected) or {times[i] for i in indices} != expected:
        raise ValueError("Weather forecast does not cover the game window")
    values = {}
    for key in VARIABLES:
        series = hourly.get(key, [])
        selected = [series[i] if i < len(series) else None for i in indices]
        if any(value is None or not math.isfinite(float(value)) for value in selected):
            raise ValueError(f"Missing hourly {key}")
        values[key] = [float(value) for value in selected]
    return {
        "temperature_c_mean": sum(values["temperature_2m"]) / len(indices),
        "wind_kmh_max": max(values["wind_speed_10m"]),
        "gust_kmh_max": max(values["wind_gusts_10m"]),
        "precipitation_mm_sum": sum(values["precipitation"]),
        "window_start": start.isoformat(),
        "hours": len(indices),
    }


def capture_weather(
    games: list[dict],
    venues: dict,
    root: Path,
    raw_root: Path,
    *,
    session=None,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(UTC)
    session = session or requests.Session()
    run = f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    rows = []
    for game in games:
        kickoff = _game_kickoff(game)
        if kickoff is None or not now < kickoff <= now + timedelta(days=14):
            continue
        venue = venues.get(str(game.get("stadium_id")))
        row = {
            "game_id": game["game_id"],
            "kickoff": kickoff.isoformat(),
            "stadium": game.get("stadium"),
            "stadium_id": game.get("stadium_id"),
            "applied_to_model": False,
            "roof": game.get("roof"),
            "status": "unavailable",
        }
        rows.append(row)
        if not venue or game.get("stadium") not in venue.get("names", []):
            row["reason"] = "unverified_venue_name_or_coordinates"
            continue
        if not (-90 <= venue["latitude"] <= 90 and -180 <= venue["longitude"] <= 180):
            row["reason"] = "invalid_coordinates"
            continue
        row.update(
            venue_source=venue["source"], latitude=venue["latitude"], longitude=venue["longitude"]
        )
        # Archive outdoor conditions even when a roof is closed. Roof status is
        # separate context: no outdoor forecast should automatically affect an indoor game.
        row["exposure"] = (
            "indoor"
            if game.get("roof") in {"dome", "closed"}
            else "outdoor"
            if game.get("roof") in {"outdoors", "open"}
            else "unknown_roof"
        )
        try:
            response = session.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": venue["latitude"],
                    "longitude": venue["longitude"],
                    "hourly": ",".join(VARIABLES),
                    "timezone": "UTC",
                    "forecast_days": 16,
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "precipitation_unit": "mm",
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            captured = datetime.now(UTC)
            atomic_write_json(
                raw_root / run / f"{game['game_id']}.json",
                {"captured_at": captured.isoformat(), "response": payload},
            )
            row.update(summarize_weather(payload, kickoff))
            row.update(
                status="captured",
                captured_at=captured.isoformat(),
                provider="Open-Meteo",
                issued_at=None,
                timing_note="Live blended forecast; capture time is known, individual model issuance is not supplied",
            )
            if captured >= kickoff:
                row.update(status="ineligible", reason="captured_after_kickoff")
        except (requests.RequestException, ValueError, KeyError) as exc:
            row["reason"] = f"{type(exc).__name__}: {str(exc)[:180]}"
    path = root / f"{run}.json"
    atomic_write_json(
        path,
        {
            "created_at": now.isoformat(),
            "research_only": True,
            "attribution": "Weather data by Open-Meteo.com",
            "games": rows,
        },
    )
    return path
